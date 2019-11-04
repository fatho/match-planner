{ stdenv, lib, rustPlatform, ... }:
let
    repoPath = toString ./..;
    sources = map (s: repoPath + s) [
        "/src"
        "/src/errors.rs"
        "/src/input.rs"
        "/src/local_search.rs"
        "/src/main.rs"
        "/Cargo.toml"
        "/Cargo.lock"
    ];
    isSource = path: type: lib.lists.elem (toString path) sources;
in
    rustPlatform.buildRustPackage rec {
        name = "match-planner-${version}";
        version = "0.2.0";

        src = builtins.filterSource isSource ./..;
        cargoSha256 = "1z6b8sz3ymdjjnhx0q854lsx70izm2zvkk2q49qp6mfra02r8262";

        meta = with stdenv.lib; {
            description = "A planning program for tennis matches";
            homepage = https://github.com/fatho/match-planner;
            license = licenses.gpl3;
            maintainers = [ "Fabian Thorand" ];
            platforms = platforms.all;
        };
    }