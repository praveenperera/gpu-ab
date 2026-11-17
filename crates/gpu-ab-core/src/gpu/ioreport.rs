//! GPU performance-state residency and GPU and Neural Engine energy from IOReport
//!
//! IOReport is the private framework behind `powermetrics`, but unlike `powermetrics`
//! it needs no root. Its counters are cumulative, so each arm records raw readings
//! before and after and the difference covers the arm. These readings are only
//! recorded, never used to invalidate an arm, so any failure here gives `None` rather
//! than an error

use std::{
    ffi::{CStr, c_char, c_void},
    ptr,
    sync::OnceLock,
};

use core_foundation_sys::{
    array::{CFArrayGetCount, CFArrayGetValueAtIndex, CFArrayRef},
    base::{CFRelease, CFTypeRef, kCFAllocatorDefault},
    dictionary::{CFDictionaryGetValue, CFDictionaryRef, CFMutableDictionaryRef},
    string::{CFStringCreateWithBytes, CFStringGetCString, CFStringRef, kCFStringEncodingUTF8},
};

#[link(name = "IOReport", kind = "dylib")]
unsafe extern "C" {
    fn IOReportCopyChannelsInGroup(
        group: CFStringRef,
        subgroup: CFStringRef,
        a: u64,
        b: u64,
        c: u64,
    ) -> CFMutableDictionaryRef;
    fn IOReportMergeChannels(into: CFMutableDictionaryRef, from: CFMutableDictionaryRef, nil: CFTypeRef);
    fn IOReportCreateSubscription(
        allocator: *const c_void,
        channels: CFMutableDictionaryRef,
        subscribed: *mut CFMutableDictionaryRef,
        channel_id: u64,
        nil: CFTypeRef,
    ) -> CFTypeRef;
    fn IOReportCreateSamples(
        subscription: CFTypeRef,
        channels: CFMutableDictionaryRef,
        nil: CFTypeRef,
    ) -> CFDictionaryRef;
    fn IOReportChannelGetGroup(channel: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetChannelName(channel: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetUnitLabel(channel: CFDictionaryRef) -> CFStringRef;
    fn IOReportStateGetCount(channel: CFDictionaryRef) -> i32;
    fn IOReportStateGetResidency(channel: CFDictionaryRef, index: i32) -> i64;
    fn IOReportSimpleGetIntegerValue(channel: CFDictionaryRef, index: i32) -> i64;
}

/// Cumulative counters at one instant
#[derive(Debug, Default)]
pub(super) struct Sample {
    /// Residency of each GPU performance state, the off state first, in IOReport ticks
    pub residency: Option<Vec<i64>>,
    pub gpu_energy_j: Option<f64>,
    pub ane_energy_j: Option<f64>,
}

/// A subscription to the GPU performance states and the energy model, made once a
/// process: copying the channel lists costs about 75 ms, sampling about 5 ms
struct Subscription {
    subscription: CFTypeRef,
    channels: CFMutableDictionaryRef,
}

// SAFETY: the subscription is never changed after it is made, and gpu-ab samples only
// from the thread that runs the arms. IOReport is private and doesn't document whether
// sampling one subscription from several threads at once is safe, so don't
unsafe impl Send for Subscription {}
unsafe impl Sync for Subscription {}

static SUBSCRIPTION: OnceLock<Option<Subscription>> = OnceLock::new();

impl Subscription {
    /// Subscribes, or gives `None` when IOReport offers neither group, as in some virtual
    /// machines. The subscription lives for the rest of the process
    fn new() -> Option<Self> {
        // SAFETY: the strings and channel lists created here are released once the
        // subscription holds what it needs, and null results are checked
        unsafe {
            let group = cf_string("GPU Stats");
            let subgroup = cf_string("GPU Performance States");
            let energy = cf_string("Energy Model");
            let channels = IOReportCopyChannelsInGroup(group, subgroup, 0, 0, 0);
            let energy_channels = IOReportCopyChannelsInGroup(energy, ptr::null(), 0, 0, 0);
            for s in [group, subgroup, energy] {
                CFRelease(s.cast());
            }

            let channels = match (channels.is_null(), energy_channels.is_null()) {
                (true, true) => return None,
                (true, false) => energy_channels,
                (false, true) => channels,
                (false, false) => {
                    IOReportMergeChannels(channels, energy_channels, ptr::null());
                    CFRelease(energy_channels.cast_const().cast());
                    channels
                }
            };

            let mut subscribed: CFMutableDictionaryRef = ptr::null_mut();
            let subscription = IOReportCreateSubscription(ptr::null(), channels, &mut subscribed, 0, ptr::null());
            CFRelease(channels.cast_const().cast());
            if subscription.is_null() || subscribed.is_null() {
                return None;
            }

            Some(Self {
                subscription,
                channels: subscribed,
            })
        }
    }
}

/// Reads one sample of the GPU performance states and the energy model, or `None` when
/// IOReport can't be read
pub(super) fn sample() -> Option<Sample> {
    let sub = SUBSCRIPTION.get_or_init(Subscription::new).as_ref()?;
    // SAFETY: the subscription is alive for the whole process, the samples are released
    // after reading, and the channel dictionaries inside them are only borrowed
    unsafe {
        let samples = IOReportCreateSamples(sub.subscription, sub.channels, ptr::null());
        if samples.is_null() {
            return None;
        }

        let result = read(samples);
        CFRelease(samples.cast());
        Some(result)
    }
}

/// # Safety
///
/// `samples` must be a live dictionary returned by `IOReportCreateSamples`
unsafe fn read(samples: CFDictionaryRef) -> Sample {
    let mut out = Sample::default();
    unsafe {
        let key = cf_string("IOReportChannels");
        let list: CFArrayRef = CFDictionaryGetValue(samples, key.cast()).cast();
        CFRelease(key.cast());
        if list.is_null() {
            return out;
        }

        for i in 0..CFArrayGetCount(list) {
            let channel: CFDictionaryRef = CFArrayGetValueAtIndex(list, i).cast();
            let group = rust_string(IOReportChannelGetGroup(channel));
            let name = rust_string(IOReportChannelGetChannelName(channel));
            match (group.as_deref(), name.as_deref()) {
                (Some("GPU Stats"), Some("GPUPH")) => {
                    let n = IOReportStateGetCount(channel);
                    out.residency = Some((0..n).map(|s| IOReportStateGetResidency(channel, s)).collect());
                }
                // "GPU Energy" counts in nJ and moves every sample; the mJ "GPU" channel
                // updates too rarely for one arm. The ANE has only the mJ channel
                (Some("Energy Model"), Some("GPU Energy")) => out.gpu_energy_j = energy_j(channel),
                (Some("Energy Model"), Some("ANE")) => out.ane_energy_j = energy_j(channel),
                _ => {}
            }
        }
    }

    out
}

/// # Safety
///
/// `channel` must be a live simple-format IOReport channel
unsafe fn energy_j(channel: CFDictionaryRef) -> Option<f64> {
    let (raw, unit) = unsafe {
        (
            IOReportSimpleGetIntegerValue(channel, 0),
            rust_string(IOReportChannelGetUnitLabel(channel)),
        )
    };
    let scale = match unit.as_deref()? {
        "nJ" => 1e-9,
        "uJ" => 1e-6,
        "mJ" => 1e-3,
        "J" => 1.0,
        _ => return None,
    };
    Some(raw as f64 * scale)
}

/// # Safety
///
/// The caller must release the returned string
unsafe fn cf_string(s: &str) -> CFStringRef {
    unsafe { CFStringCreateWithBytes(kCFAllocatorDefault, s.as_ptr(), s.len() as _, kCFStringEncodingUTF8, 0) }
}

/// # Safety
///
/// `s` must be null or a live CFString
unsafe fn rust_string(s: CFStringRef) -> Option<String> {
    if s.is_null() {
        return None;
    }

    let mut buf = [0 as c_char; 256];
    // SAFETY: the buffer is writable for its whole length, which is passed along
    let ok = unsafe { CFStringGetCString(s, buf.as_mut_ptr(), buf.len() as _, kCFStringEncodingUTF8) };
    // SAFETY: CFStringGetCString wrote a NUL-terminated string when it succeeded
    (ok != 0).then(|| unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
}
