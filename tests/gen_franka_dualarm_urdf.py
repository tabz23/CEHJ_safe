"""Generate dual-arm URDFs in HoloBrain's piper_description_dualarm
convention from RoboTwin's single-arm URDFs.

Convention (mirrors piper_description_dualarm.urdf):
  - root `base_link`
  - left arm chain mounted at identity via a fixed joint
  - right arm chain mounted at (0, -0.6, 0) via a fixed joint
    (RoboTwin mounts the arms 0.6 m apart along world x; with the arm
    base's +90 deg z rotation that is (0, -0.6, 0) in the left base frame)
  - all link/joint names prefixed left_/right_
  - only kinematics (links as empty shells, joints with origin/axis/limit);
    pytorch_kinematics needs nothing else and this sidesteps mesh paths.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

ROBOTWIN = Path("/root/autodl-tmp/RoboTwin/assets/embodiments")
OUT = Path(__file__).resolve().parents[1] / "assets" / "urdf"

SPECS = {
    "franka_panda_dualarm.urdf": (ROBOTWIN / "franka-panda" / "panda.urdf",
                                  "panda_link0"),
    "arx_x5_dualarm.urdf": (ROBOTWIN / "ARX-X5" / "X5A.urdf", "base_link"),
}


def copy_joint(j: ET.Element, prefix: str, out: ET.Element):
    nj = ET.SubElement(out, "joint", {"name": prefix + j.get("name"),
                                      "type": j.get("type")})
    for tag in ("origin", "axis", "limit"):
        el = j.find(tag)
        if el is not None:
            ET.SubElement(nj, tag, dict(el.attrib))
    parent = j.find("parent")
    child = j.find("child")
    ET.SubElement(nj, "parent", {"link": prefix + parent.get("link")})
    ET.SubElement(nj, "child", {"link": prefix + child.get("link")})


def generate(src: Path, root_link: str, dst: Path):
    src_root = ET.parse(src).getroot()
    links = [el.get("name") for el in src_root.findall("link")]
    joints = src_root.findall("joint")
    assert root_link in links, f"{root_link} not in {src}"

    robot = ET.Element("robot", {"name": f"dual_arm_{dst.stem}"})
    ET.SubElement(robot, "link", {"name": "base_link"})

    for prefix, mount_xyz in (("left_", "0 0 0"), ("right_", "0 -0.6 0")):
        for name in links:
            ET.SubElement(robot, "link", {"name": prefix + name})
        mount = ET.SubElement(robot, "joint", {
            "name": f"base_link_to_{prefix}{root_link}", "type": "fixed"})
        ET.SubElement(mount, "origin", {"xyz": mount_xyz, "rpy": "0 0 0"})
        ET.SubElement(mount, "parent", {"link": "base_link"})
        ET.SubElement(mount, "child", {"link": prefix + root_link})
        for j in joints:
            copy_joint(j, prefix, robot)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(dst, xml_declaration=True, encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    for fname, (src, root_link) in SPECS.items():
        generate(src, root_link, OUT / fname)
