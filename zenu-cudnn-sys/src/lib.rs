#![expect(non_upper_case_globals)]
#![expect(non_camel_case_types)]
#![expect(non_snake_case)]
#![expect(clippy::unreadable_literal)]
#![expect(clippy::pub_underscore_fields)]

// include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
include!("bindings.rs");
