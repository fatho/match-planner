let
    pkgs = import ./nix/nixpkgs-pinned.nix {
        config = {};
        overlays = [
            (import ./nix/overlay.nix)
        ];
    };
in
    pkgs.match-planner