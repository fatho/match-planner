let
    mozilla = import ./nixpkgs-mozilla-pinned.nix;
    pkgs = import ./nixpkgs-pinned.nix { overlays = [mozilla]; };

    rustChannel = pkgs.rustChannels.stable;

    # Findings:
    # - rustc uses default ld (via cc) for making `rlibs` for crates
    # - default ld is confused when -lpthread is in path in NIX_LDFLAGS
    # - when making a static library (such as rlibs), ld presumably ignores the `-l` libraries, unless it cannot even parse them
    # - default ld cannot parse the windows pthread library

    # - dependencies must be compiled without phtreads
    # - match-planner itself must be compiled with pthreads
in
    pkgs.mkShell {
        name = "match-planner";
        nativeBuildInputs = [
            # gcc is needed for linking
            pkgs.pkgsCross.mingwW64.buildPackages.gcc
            (rustChannel.rust.override {
                targets = [ "x86_64-pc-windows-gnu" ];
                extensions = [ "rust-std" ];
            })

            # Utils for experimenting in the shell
            pkgs.which
        ];
        buildInputs = [
            pkgs.pkgsCross.mingwW64.windows.pthreads
        ];
    }
