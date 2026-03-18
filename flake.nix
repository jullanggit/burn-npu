{
  description = "Development shell for burn-npu";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.openssl
            pkgs.openvino
            pkgs.level-zero
            pkgs.vulkan-loader
            pkgs.libglvnd
          ];

          packages = with pkgs; [
            pkg-config
            openssl
            openvino
            level-zero
            intel-npu-driver
            vulkan-loader
            vulkan-headers
            vulkan-tools
            libglvnd
          ];
        };
      }
    );
}
