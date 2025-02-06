import xml.etree.ElementTree as ET
import os
import random
script_directory = os.path.dirname(os.path.abspath(__file__))

tree = ET.parse(script_directory + "/rou/d_1_turnCount_am_offpeak_1.rou.xml")

root = tree.getroot()
xml = """
<?xml version="1.0" encoding="UTF-8"?>

<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
{routes}
{vehicles}
</routes>
""".strip()
with open(script_directory + "/rou/generated/random.rou.xml", "w+") as f:
    all_routes = ""
    for route in root.findall("route"):
        route_str = ET.tostring(route, encoding="unicode")
        all_routes += "\n\t" + route_str.strip()
    
    all_vehicles = ""
    
    for vehicle in root.findall("vehicle"):
        vehicle_str = ET.tostring(vehicle, encoding="unicode")
        if random.uniform(0, 1) > 0.75:
            all_vehicles += "\n\t" + vehicle_str.strip()
    
    f.write(xml.format(routes=all_routes, vehicles=all_vehicles))