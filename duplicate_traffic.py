import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

# Define the input and output file names
input_file_name = "rou/d_1_turnCount_am_offpeak_1.rou.xml"  # Replace with your actual file name
output_file_name = "rou/d_1_turnCount_am_offpeak_1_duplicated.rou.xml"

# Parse the XML file and get the root element
tree = ET.parse(input_file_name)
root = tree.getroot()

# Calculate the maximum departure time
max_time = max(float(vehicle.get("depart")) for vehicle in root.findall("vehicle"))

# Find the maximum ID to ensure unique duplication
max_id = max(int(vehicle.get("id")) for vehicle in root.findall("vehicle"))

# Create duplicated vehicles with updated departure times
for vehicle in list(root):  # Use a list copy to prevent modifying the element during iteration
    if vehicle.tag == "vehicle":  # Check if the tag is a vehicle
        # Create a new vehicle element based on the current vehicle's attributes
        new_vehicle = ET.Element("vehicle", vehicle.attrib)
        
        # Update the id and departure time for the new vehicle
        new_vehicle.set("id", str(int(vehicle.get("id")) + max_id + 1))  # Assign a new id
        new_vehicle.set("depart", str(round(float(vehicle.get("depart")) + max_time, 2)))  # Add max_time to departure
        
        # Append the duplicated vehicle to the root
        root.append(new_vehicle)

# Convert the modified XML into a pretty-printed format with proper new lines and indentation
xml_str = ET.tostring(root, encoding='unicode')  # Convert the tree to a string
formatted_xml = minidom.parseString(xml_str).toprettyxml(indent=" ")  # Pretty-print with indentation

# Write the formatted XML to the output file
with open(output_file_name, "w", encoding="utf-8") as f:
    f.write(formatted_xml)

print(f"Modified XML saved successfully as '{output_file_name}' with formatted output.")
