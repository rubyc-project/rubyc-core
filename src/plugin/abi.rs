//! Plugin ABI boundary types.
//!
//! The byte contract between the host binary and a target plugin:
//! every call passes a `*const c_char` pointer + length (a `&[u8]`), and
//! results are returned as a `*mut PluginBytes` that the caller owns and
//! must free with [`PluginBytes::release`].
//!
//! The layout is a 32-byte `#[repr(C)]` struct (with the 4-byte pad C layout
//! inserts between the 4-byte `c_int` and the first 8-byte field) so it can
//! cross the FFI boundary without alignment surprises:
//!
//! ```text
//! offset:  0          4      8          16         24         32
//!          +----------+------+----------+----------+----------+
//!          |   ok     | pad  |   len    |   cap    |  data    |
//!          |  c_int   | (4)  |   u64    |   u64    | *mut c_char|
//!          +----------+------+----------+----------+----------+
//! ```
//!
//! When `ok` is 0 the payload is a NUL-terminated UTF-8 error message;
//! when `ok` is 1 the payload is the raw result bytes.

use std::os::raw::{c_char, c_int};

/// A byte buffer that crosses the FFI boundary.
#[repr(C)]
pub struct PluginBytes {
    pub ok: c_int,
    pub len: u64,
    pub cap: u64,
    pub data: *mut c_char,
}

impl PluginBytes {
    /// An empty, valid buffer.
    pub fn empty() -> Self {
        Self {
            ok: 1,
            len: 0,
            cap: 0,
            data: std::ptr::null_mut(),
        }
    }

    /// A successful result carrying `data`.
    pub fn success(data: Vec<u8>) -> Self {
        let cap = data.len() as u64;
        let len = data.len() as u64;
        let ptr = data.as_ptr() as *mut c_char;
        // Leak the Vec so the buffer lives until release().
        std::mem::forget(data);
        Self {
            ok: 1,
            len,
            cap,
            data: ptr,
        }
    }

    /// A failure carrying an NUL-terminated message.
    pub fn error(msg: String) -> Self {
        let mut v = msg.into_bytes();
        v.push(0);
        let cap = v.len() as u64;
        let len = v.len() as u64;
        let ptr = v.as_ptr() as *mut c_char;
        std::mem::forget(v);
        Self {
            ok: 0,
            len,
            cap,
            data: ptr,
        }
    }

    /// Free the buffer. Safe to call on a null/empty buffer.
    pub unsafe fn release(&self) {
        if self.data.is_null() || self.cap == 0 {
            return;
        }
        unsafe {
            let _ = Vec::from_raw_parts(self.data as *mut u8, self.len as usize, self.cap as usize);
        }
    }
}

impl Drop for PluginBytes {
    fn drop(&mut self) {
        if !self.data.is_null() && self.cap != 0 {
            unsafe {
                let _ = Vec::from_raw_parts(self.data as *mut u8, self.len as usize, self.cap as usize);
            }
        }
    }
}

/// Copy a `PluginBytes` result into a `Vec<u8>`, or return the error message
/// as a `String` on failure.
pub fn take_result(b: PluginBytes) -> Result<Vec<u8>, String> {
    if b.ok != 0 {
        let mut v = Vec::with_capacity(b.len as usize);
        if !b.data.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(b.data as *const u8, v.as_mut_ptr(), b.len as usize);
                v.set_len(b.len as usize);
            }
        }
        Ok(v)
    } else {
        let mut s = String::new();
        if !b.data.is_null() {
            unsafe {
                let mut i = 0;
                while *b.data.add(i) != 0 {
                    s.push(*b.data.add(i) as u8 as char);
                    i += 1;
                }
            }
        }
        Err(s)
    }
}
