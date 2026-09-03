use fst::MapBuilder;
use net::hsts::HstsPreloadList;
use servo as _;

#[test]
fn preload_lookup_enforces_exact_and_include_subdomain_entries() {
    let mut builder = MapBuilder::memory();
    builder.insert("example.com", 0).unwrap();
    builder.insert("example.org", 3).unwrap();
    let preload = HstsPreloadList(builder.into_map());

    assert!(!preload.is_host_secure("derp"));
    assert!(preload.is_host_secure("example.com"));
    assert!(!preload.is_host_secure("servo.example.com"));
    assert!(preload.is_host_secure("example.org"));
    assert!(preload.is_host_secure("servo.example.org"));
}
