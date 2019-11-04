# Standalone derivation for making a windows build of match-planner
let
    mozilla = import ./nix/nixpkgs-mozilla-pinned.nix;
    pkgs = import ./nix/nixpkgs-pinned.nix {
        config = {};
        overlays = [
            mozilla
            (import ./nix/overlay.nix)
        ];
    };

    rustChannel = pkgs.rustChannels.stable;

    rust = rustChannel.rust.override {
        targets = [ "x86_64-pc-windows-gnu" ];
        extensions = [ "rust-std" ];
    };
in
    pkgs.pkgsCross.mingwW64.match-planner.override {
        # Prevent Nix from compiling rustc from scratch for mingw, which it by
        # default does for compilers. However, unlike gcc, rustc is perfectly
        # capable of compiling towards different targets without recompiling
        # the compiler.
        rustPlatform.buildRustPackage = pkgs.pkgsCross.mingwW64
            .rustPlatform.buildRustPackage.override {
                rustc = rust;
                cargo = rust;
        };
    }