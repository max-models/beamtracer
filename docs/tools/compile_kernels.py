"""Compile the Pyccel kernels before the docs notebooks run, failing if one cannot be.

Without this a kernel that fails to compile runs its NumPy version without a word, which
makes the optimisation tutorials take hours instead of minutes.
"""

from beamtracer.kernels.catalog import KERNELS


def main() -> None:
    """Build every host kernel and raise unless all of them are the Pyccel version."""
    failed = {}
    for name, kernel in KERNELS.items():
        implementations = kernel.host_kernel.kernel
        chosen = implementations.build()
        print(f"{name}: {chosen}")
        if chosen != "pyccel":
            failed[name] = implementations.errors
    if failed:
        raise SystemExit(f"kernels not compiled with Pyccel: {failed}")


if __name__ == "__main__":
    main()
