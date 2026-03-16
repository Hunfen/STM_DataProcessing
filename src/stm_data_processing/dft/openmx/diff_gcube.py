#!/usr/bin/env python3
"""
diff_gcube.py - Calculate the difference between two Gaussian cube files.

Usage:
    python diff_gcube.py input1.cube input2.cube output.cube

Definition of difference:
    input1 - input2 = output
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self


@dataclass
class CubeFile:
    """Represents a Gaussian cube file."""

    comment1: str
    comment2: str
    atom_num: int
    origin: list[float]
    ngrid: list[int]
    gtv: list[list[float]]
    atoms: list[list[float]]
    data: list[list[list[float]]]

    @classmethod
    def read(cls, filepath: str | Path) -> Self:
        """Read a Gaussian cube file."""
        path = Path(filepath)
        with path.open("r") as f:
            lines = f.readlines()

        comment1 = lines[0].strip()
        comment2 = lines[1].strip()

        # Line 3: atom_num, origin_x, origin_y, origin_z
        parts = lines[2].split()
        atom_num = int(parts[0])
        origin = [float(x) for x in parts[1:4]]

        # Lines 4-6: grid info
        ngrid = []
        gtv = []
        for i in range(3):
            parts = lines[3 + i].split()
            ngrid.append(int(parts[0]))
            gtv.append([float(x) for x in parts[1:4]])

        # Atom coordinates (lines 7 to 7+atom_num-1)
        atoms = []
        for i in range(atom_num):
            parts = lines[6 + i].split()
            atoms.append([float(x) for x in parts[:5]])

        # Grid data (remaining lines)
        data = [[[0.0] * ngrid[2] for _ in range(ngrid[1])] for _ in range(ngrid[0])]
        data_start_line = 6 + atom_num
        data_lines = lines[data_start_line:]

        # Parse data values
        values = []
        for line in data_lines:
            values.extend(float(x) for x in line.split())

        # Fill 3D array
        idx = 0
        for n1 in range(ngrid[0]):
            for n2 in range(ngrid[1]):
                for n3 in range(ngrid[2]):
                    data[n1][n2][n3] = values[idx]
                    idx += 1

        return cls(
            comment1=comment1,
            comment2=comment2,
            atom_num=atom_num,
            origin=origin,
            ngrid=ngrid,
            gtv=gtv,
            atoms=atoms,
            data=data,
        )

    def write(self, filepath: str | Path) -> None:
        """Write to a Gaussian cube file."""
        path = Path(filepath)
        with path.open("w") as f:
            f.write(f"{self.comment1}\n")
            f.write(f"{self.comment2}\n")
            f.write(
                f"{self.atom_num:5d} {self.origin[0]:12.6f} "
                f"{self.origin[1]:12.6f} {self.origin[2]:12.6f}\n"
            )
            for i in range(3):
                f.write(
                    f"{self.ngrid[i]:5d} {self.gtv[i][0]:12.6f} "
                    f"{self.gtv[i][1]:12.6f} {self.gtv[i][2]:12.6f}\n"
                )

            for atom in self.atoms:
                f.write(
                    f"{atom[0]:5.0f} {0.0:12.6f} {atom[1]:12.6f} "
                    f"{atom[2]:12.6f} {atom[3]:12.6f}\n"
                )

            # Write grid data (6 values per line)
            for n1 in range(self.ngrid[0]):
                for n2 in range(self.ngrid[1]):
                    for n3 in range(self.ngrid[2]):
                        f.write(f"{self.data[n1][n2][n3]:13.3E}")
                        if (n3 + 1) % 6 == 0:
                            f.write("\n")
                    if self.ngrid[2] % 6 != 0:
                        f.write("\n")

    def validate_compatibility(self, other: Self) -> None:
        """Validate that two cube files are compatible for subtraction."""
        errors = []

        if self.ngrid != other.ngrid:
            errors.append("Found a difference in the number of grid")

        if self.origin != other.origin:
            errors.append("Found a difference in the origin coordinates")

        if self.gtv != other.gtv:
            errors.append("Found a difference in the grid vectors")

        if errors:
            for err in errors:
                print(err)
            raise ValueError("Cube files are not compatible for subtraction")


def diff_cube_files(input1: str | Path, input2: str | Path, output: str | Path) -> None:
    """
    Calculate the difference between two cube files.

    Args:
        input1: Path to the first cube file (minuend).
        input2: Path to the second cube file (subtrahend).
        output: Path to the output cube file (difference).
    """
    # Read both files
    cube1 = CubeFile.read(input1)
    cube2 = CubeFile.read(input2)

    # Validate compatibility
    cube1.validate_compatibility(cube2)

    # Calculate difference: input1 - input2
    diff_data = [
        [
            [
                cube1.data[n1][n2][n3] - cube2.data[n1][n2][n3]
                for n3 in range(cube1.ngrid[2])
            ]
            for n2 in range(cube1.ngrid[1])
        ]
        for n1 in range(cube1.ngrid[0])
    ]

    # Create output cube file
    output_cube = CubeFile(
        comment1=cube1.comment1,
        comment2=cube1.comment2,
        atom_num=cube1.atom_num,
        origin=cube1.origin,
        ngrid=cube1.ngrid,
        gtv=cube1.gtv,
        atoms=cube1.atoms,
        data=diff_data,
    )

    # Write output
    output_cube.write(output)
    print(f"Successfully wrote difference to {output}")


def main() -> None:
    """Main entry point."""
    import sys

    if len(sys.argv) != 4:
        print("Usage:")
        print("  python diff_gcube.py input1.cube input2.cube output.cube")
        sys.exit(1)

    input1, input2, output = sys.argv[1], sys.argv[2], sys.argv[3]
    diff_cube_files(input1, input2, output)


if __name__ == "__main__":
    main()
