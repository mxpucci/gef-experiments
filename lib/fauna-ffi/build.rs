use std::{env, fs, path::PathBuf};

fn main() {
    let source = PathBuf::from(env::var_os("GEF_FAUNA_UPSTREAM").expect("GEF_FAUNA_UPSTREAM missing"));
    let source = source.join("crates/modelardb_compression/src");
    let mut modules = String::new();
    for (name, file) in [("error", "error.rs"), ("models", "models/mod.rs"),
                         ("types", "types.rs"), ("compression", "compression.rs")] {
        let path = source.join(file);
        modules.push_str(&format!("#[path = {:?}] mod {name};\n", path));
    }
    fs::write(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("upstream.rs"), modules).unwrap();
    println!("cargo:rerun-if-env-changed=GEF_FAUNA_UPSTREAM");
}
