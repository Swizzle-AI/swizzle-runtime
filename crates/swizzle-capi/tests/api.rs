//! The C entry points, called as C would call them.
//!
//! SWIZZLE_TEST_GGUF=SmolLM2-135M-Instruct-f16.gguf SWIZZLE_TEST_EMBED_GGUF=bge-small-en-v1.5-f16.gguf
//! cargo test --release -p swizzle-capi -- --include-ignored

use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

use swizzle_capi::*;

fn last_error() -> String {
    let e = swizzle_last_error();
    assert!(!e.is_null());
    unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
}

fn env_path(var: &str) -> CString {
    CString::new(std::env::var(var).unwrap_or_else(|_| panic!("set {var}"))).unwrap()
}

extern "C" fn collect(text: *const c_char, len: usize, out: *mut c_void) -> bool {
    let out = unsafe { &mut *out.cast::<String>() };
    let piece = unsafe { CStr::from_ptr(text) }.to_str().unwrap();
    assert_eq!(piece.len(), len);
    out.push_str(piece);
    true
}

#[test]
fn version_is_the_crate_version() {
    let v = unsafe { CStr::from_ptr(swizzle_version()) };
    assert_eq!(v.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn null_and_missing_inputs_fail_with_a_reason() {
    unsafe {
        assert!(swizzle_model_load(ptr::null()).is_null());
        assert_eq!(last_error(), "path is NULL");
        let missing = CString::new("no-such-model.gguf").unwrap();
        assert!(swizzle_model_load(missing.as_ptr()).is_null());
        assert!(last_error().contains("no-such-model.gguf"));
        assert!(swizzle_session_new(ptr::null(), ptr::null()).is_null());
        let none = swizzle_session_generate(
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            None,
            ptr::null_mut(),
        );
        assert_eq!(none, -1);
        assert_eq!(swizzle_embedding_size(ptr::null()), 0);
        swizzle_model_free(ptr::null_mut());
        swizzle_session_free(ptr::null_mut());
    }
}

#[test]
#[ignore = "needs SWIZZLE_TEST_GGUF"]
fn a_session_remembers_its_turns() {
    unsafe {
        let model = swizzle_model_load(env_path("SWIZZLE_TEST_GGUF").as_ptr());
        assert!(!model.is_null(), "{}", last_error());
        let system = CString::new("Answer in one short sentence.").unwrap();
        let session = swizzle_session_new(model, system.as_ptr());
        // The session keeps the model alive.
        swizzle_model_free(model);

        let mut params = swizzle_params_default();
        params.max_tokens = 12;
        params.temperature = 0.0;
        let mut first = String::new();
        let question = CString::new("What is the capital of France?").unwrap();
        let n = swizzle_session_generate(
            session,
            question.as_ptr(),
            &params,
            Some(collect),
            (&mut first as *mut String).cast(),
        );
        assert!(n > 0 && n <= 12, "{n}: {}", last_error());
        assert!(!first.trim().is_empty());

        let follow_up = CString::new("And of Italy?").unwrap();
        let n =
            swizzle_session_generate(session, follow_up.as_ptr(), &params, None, ptr::null_mut());
        assert!(n > 0, "{}", last_error());
        swizzle_session_free(session);
    }
}

#[test]
#[ignore = "needs SWIZZLE_TEST_EMBED_GGUF"]
fn embeddings_fill_the_buffer() {
    unsafe {
        let model = swizzle_model_load(env_path("SWIZZLE_TEST_EMBED_GGUF").as_ptr());
        assert!(!model.is_null(), "{}", last_error());
        assert!(swizzle_session_new(model, ptr::null()).is_null());
        let size = swizzle_embedding_size(model);
        assert!(size > 0);
        let mut out = vec![f32::NAN; size];
        let text = CString::new("a cat on a mat").unwrap();
        assert_eq!(swizzle_embed(model, text.as_ptr(), out.as_mut_ptr()), 0);
        assert!(out.iter().all(|x| x.is_finite()));
        assert_eq!(swizzle_embed(model, text.as_ptr(), ptr::null_mut()), -1);
        swizzle_model_free(model);
    }
}
