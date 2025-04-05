###########################################
#
# combine.py
#
# Takes the 3 intermediate ply files output from 
# SegmentAnyTree and combines them into a single 
# compressed lidar file (*.LAZ).   Replaces 
# the pandas_to_las.py script in SegmentAnyTree
#
# Luke Sheneman
# Institute for Interdisciplinary Data Sciences (IIDS)
# sheneman@uidaho.edu
# April 2025
#
#
############################################

import argparse
import numpy as np
import laspy
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys
import os
import time 


# reads specific properties from a PLY file into a dictionary of numpy arrays.
def read_ply_data(filepath, required_props):
    print(f"Reading PLY: {filepath}")
    start_time = time.time()
    try:
        plydata = PlyData.read(filepath)
        vertex_element = plydata['vertex']
        data = {}
        print(f"  Properties in file: {vertex_element.properties}")
        available_props = [p.name for p in vertex_element.properties]

        found_props = []
        missing_props = []

        for prop in required_props:
            # Handle case variations common in PLY files vs LAS standard
            prop_found = None
            if prop in available_props:
                prop_found = prop
            elif prop.lower() in [p.lower() for p in available_props]:
                 # Find the actual case-sensitive name
                 prop_found = next((p for p in available_props if p.lower() == prop.lower()), None)

            if prop_found:
                data[prop] = vertex_element[prop_found] # Store with requested name
                found_props.append(prop)
            else:
                print(f"  Warning: Property '{prop}' not found in {filepath}.")
                missing_props.append(prop)

        num_points = len(vertex_element.data)
        print(f"  Read {num_points} vertices and properties: {found_props} in {time.time() - start_time:.2f}s.")
        return data, num_points, missing_props
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading PLY file {filepath}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



# replaces NaN/inf in an array and casts to integer type.
def clean_int_array(arr, target_dtype, fill_value=0, verbose=False, col_name=""):
    if not np.issubdtype(arr.dtype, np.number): # Skip if not numeric
        if verbose: print(f"  Skipping NaN/inf check for non-numeric column: {col_name}")
        return arr.astype(target_dtype) # Still attempt cast

    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    has_nan = np.any(nan_mask)
    has_inf = np.any(inf_mask)

    if has_nan or has_inf:
        if verbose: print(f"  Cleaning NaNs/Infs in column: {col_name}")
        # Work on a copy if it's not already a float (to allow NaN representation)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(float)
        arr[inf_mask | nan_mask] = fill_value # Fill NaNs and Infs
        # Check bounds for the fill_value
        try:
            dtype_info = np.iinfo(target_dtype)
            if fill_value < dtype_info.min or fill_value > dtype_info.max:
                 print(f"  Warning: Fill value {fill_value} is out of bounds for {col_name} ({target_dtype}). Check your data or fill value.")
                 # Optionally clamp: arr = np.clip(arr, dtype_info.min, dtype_info.max)
        except ValueError:
            pass # Not an integer type info (shouldn't happen here)

    # Cast to final integer type
    try:
        return arr.astype(target_dtype)
    except ValueError as e:
         print(f"Error casting column '{col_name}' to {target_dtype} after cleaning: {e}")
         # Consider alternative handling, like saving as float if essential
         raise e




def main(base_ply_path, semantic_ply_path, instance_ply_path, output_las_path, compress=True):

    #########################
    # Read Base Point Cloud # 
    #########################

    # Define properties we absolutely need for LAS coordinates
    base_coord_props = ['X', 'Y', 'Z']
    # Define standard LAS properties we *want* to copy if they exist
    base_standard_props = ['intensity', 'return_number', 'number_of_returns',
                           'scan_direction_flag', 'edge_of_flight_line', 'classification',
                           'synthetic', 'key_point', 'withheld', 'scan_angle_rank', # will be renamed
                           'user_data', 'point_source_id', 'gps_time',
                           'red', 'green', 'blue']
    # Define potential other properties in the PLY we might want as extra dims
    base_extra_props = ['hag', 'RelativeReflectivity', 'snr', 'Amplitude', 'Reflectance', 'Deviation', 'Pulse_width', 'overlap', 'scanner_channel'] # Add others from your header if needed

    base_props_to_read = base_coord_props + base_standard_props + base_extra_props
    base_data_dict, num_base_points, _ = read_ply_data(base_ply_path, base_props_to_read)

    if num_base_points == 0: sys.exit("Error: Base PLY file has no points.")
    if not all(prop in base_data_dict for prop in base_coord_props):
        sys.exit(f"Error: Base PLY file missing one or more coordinate properties ({base_coord_props}).")

    # Store coordinates separately for clarity and KNN
    base_coords = np.vstack((base_data_dict['X'], base_data_dict['Y'], base_data_dict['Z'])).T



    ##############################
    # Read Semantic Segmentation #
    ##############################

    semantic_props_to_read = ['preds']
    semantic_data_dict, num_semantic_points, _ = read_ply_data(semantic_ply_path, semantic_props_to_read)

    if num_semantic_points != num_base_points:
        sys.exit(f"Error: Semantic PLY point count ({num_semantic_points}) differs from base PLY ({num_base_points}).")
    if 'preds' not in semantic_data_dict:
        sys.exit("Error: 'preds' column not found in semantic PLY.")

    # Assign semantic predictions (will be added as extra dim later)
    pred_semantic = semantic_data_dict['preds']
    print("Semantic predictions read.")




    ##############################
    # Read Instance Segmentation #
    ##############################

    instance_props_to_read = ['x', 'y', 'z', 'preds']
    instance_data_dict, num_instance_points, _ = read_ply_data(instance_ply_path, instance_props_to_read)

    if num_instance_points == 0: sys.exit("Error: Instance PLY file has no points.")
    if not all(prop in instance_data_dict for prop in instance_props_to_read):
         sys.exit(f"Error: Instance PLY file missing one or more required properties ({instance_props_to_read}).")

    instance_coords = np.vstack((instance_data_dict['x'], instance_data_dict['y'], instance_data_dict['z'])).T
    instance_labels = instance_data_dict['preds']
    print(f"Instance data read: {instance_coords.shape[0]} points.")


    ############################
    # Nearest-Neighbor mapping #
    ############################
    print("Building KDTree for instance points...")
    start_knn = time.time()
    kdtree = cKDTree(instance_coords)
    print(f"KDTree built in {time.time() - start_knn:.2f}s.")

    print(f"Querying KDTree for {num_base_points} base points...")
    start_query = time.time()
    # Find the index of the single closest point in instance_coords for each base_coord
    distances, indices = kdtree.query(base_coords, k=1, workers=-1) # Use all available cores
    print(f"KDTree query finished in {time.time() - start_query:.2f}s.")

    # Map instance labels (handle potential out-of-bounds indices if query failed, though unlikely with k=1)
    valid_indices_mask = indices < len(instance_labels)
    pred_instance = np.full(num_base_points, 0, dtype=instance_labels.dtype) # Default to 0
    pred_instance[valid_indices_mask] = instance_labels[indices[valid_indices_mask]]

    num_failed_map = num_base_points - np.sum(valid_indices_mask)
    if num_failed_map > 0:
        print(f"Warning: {num_failed_map} base points could not be mapped to an instance label (KNN index out of bounds). They will have instance ID 0.")

    print("Instance predictions mapped using KNN.")



    ####################
    # Build LAS Header #
    ####################
    print("Setting up LAS header...")
    # Choose point format based on available standard fields
    has_gps_time = 'gps_time' in base_data_dict
    has_rgb = all(prop in base_data_dict for prop in ['red', 'green', 'blue'])
    # Add check for NIR if needed for format 8/10
    # has_nir = 'nir' in base_data_dict

    point_format_id = 6 # Default with time
    if has_rgb:
        point_format_id = 7 # Format 7 has Time + RGB
    elif has_gps_time:
        point_format_id = 6 # Format 6 has Time
    else:
        point_format_id = 3 # Basic XYZIRNCS + Scan Angle (if available) - adjust if needed
        # Or maybe default to 2 if no RGB and no time? Check LAS spec vs your data needs.

    print(f"  Selected Point Format ID: {point_format_id}")
    header = laspy.LasHeader(point_format=point_format_id, version="1.4")

    # Define data types for standard LAS fields (use numpy types)
    # Match these to the Point Format definition
    las_standard_dtypes = {
        'intensity': np.uint16, 'return_number': np.uint8, 'number_of_returns': np.uint8,
        'scan_direction_flag': bool, 'edge_of_flight_line': bool, 'classification': np.uint8,
        'synthetic': bool, 'key_point': bool, 'withheld': bool,
        'scan_angle': np.int16, # scan_angle_rank maps to this scaled field
        'user_data': np.uint8, 'point_source_id': np.uint16, 'gps_time': np.float64,
        'red': np.uint16, 'green': np.uint16, 'blue': np.uint16
        # Add 'nir': np.uint16 if using formats 8 or 10
    }

    # Define extra dimensions to add
    extra_dims_to_add = {}
    # Add semantic prediction
    extra_dims_to_add['PredSemantic'] = np.uint8 # Ensure this matches expected output type
    # Add instance prediction
    extra_dims_to_add['PredInstance'] = np.uint16 # Ensure this matches expected output type

    # Add any other non-standard fields from the base PLY
    for prop in base_extra_props:
        if prop in base_data_dict:
             # Try to infer numpy dtype, default to float64 if unknown
             dtype = base_data_dict[prop].dtype
             if dtype.kind not in 'biuf': # bool, int, uint, float
                  print(f"  Warning: Non-standard dtype '{dtype}' for extra dim '{prop}'. Defaulting to float64.")
                  dtype = np.float64
             extra_dims_to_add[prop] = dtype

    # Add extra dimension definitions to header
    for name, dtype in extra_dims_to_add.items():
        try:
             print(f"  Adding Extra Dim: {name} ({dtype})")
             header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        except Exception as e:
             print(f"  Warning: Failed to add extra dim '{name}'. Error: {e}")


    # Set scale, offset, bounds (using original float coords before scaling)
    header.offsets = np.min(base_coords, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001]) # Use appropriate scales
    header.mins = np.min(base_coords, axis=0)
    header.maxs = np.max(base_coords, axis=0)



    #####################
    # Populate LAS Data #
    #####################
    print("Populating LAS data...")
    las = laspy.LasData(header)

    las.x = base_coords[:, 0]
    las.y = base_coords[:, 1]
    las.z = base_coords[:, 2]

    # Populate standard dimensions
    for las_name, np_type in las_standard_dtypes.items():
        # Handle scan_angle_rank rename
        ply_name = 'scan_angle_rank' if las_name == 'scan_angle' else las_name

        if ply_name in base_data_dict:
            print(f"  Assigning standard dimension: {las_name}")
            data_array = base_data_dict[ply_name]
            # Clean and cast if it's an integer type
            if 'int' in str(np_type):
                fill_val = -9999 if np_type in [np.int8, np.int16, np.int32, np.int64] else 0
                data_array = clean_int_array(data_array, np_type, fill_value=fill_val, verbose=True, col_name=las_name)
            else:
                # For floats/bools, just ensure type consistency
                data_array = data_array.astype(np_type)
            setattr(las, las_name, data_array)
        # else: # Optionally fill with defaults if missing, but laspy often handles this
        #     print(f"  Standard dimension '{las_name}' not found in source, skipping or using default.")
        #     pass


    # Populate extra dimensions
    print(f"  Assigning extra dimension: PredSemantic")
    las.PredSemantic = clean_int_array(pred_semantic, extra_dims_to_add['PredSemantic'], fill_value=0, verbose=True, col_name='PredSemantic') # uint8, use 0

    print(f"  Assigning extra dimension: PredInstance")
    las.PredInstance = clean_int_array(pred_instance, extra_dims_to_add['PredInstance'], fill_value=0, verbose=True, col_name='PredInstance') # uint16, use 0

    for name, dtype in extra_dims_to_add.items():
        if name in ['PredSemantic', 'PredInstance']: continue # Already handled

        if name in base_data_dict:
            print(f"  Assigning extra dimension: {name}")
            data_array = base_data_dict[name]
            if 'int' in str(dtype):
                fill_val = -9999 if dtype in [np.int8, np.int16, np.int32, np.int64] else 0
                data_array = clean_int_array(data_array, dtype, fill_value=fill_val, verbose=True, col_name=name)
            else:
                data_array = data_array.astype(dtype) # Ensure correct float/bool type
            setattr(las, name, data_array)



    #######################
    #  Write LAS/LAZ File #
    #######################
    output_dir = os.path.dirname(output_las_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    write_path = output_las_path
    if compress:
        if not write_path.lower().endswith('.laz'):
            write_path = os.path.splitext(write_path)[0] + '.laz'
        print(f"Writing compressed LAZ file to: {write_path}")
        las.write(write_path)
    else:
        if not write_path.lower().endswith('.las'):
             write_path = os.path.splitext(write_path)[0] + '.las'
        print(f"Writing uncompressed LAS file to: {write_path}")
        las.write(write_path)

    print(f"Successfully created file: {write_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine base PLY, semantic PLY, and instance PLY into a single LAS/LAZ file.")
    parser.add_argument("base_ply", help="Path to the base point cloud PLY file.")
    parser.add_argument("semantic_ply", help="Path to the semantic segmentation PLY file.")
    parser.add_argument("instance_ply", help="Path to the instance segmentation PLY file.")
    parser.add_argument("output_las", help="Path for the output LAS/LAZ file.")
    parser.add_argument("--no-compress", action="store_true", help="Save as uncompressed LAS instead of LAZ.")

    args = parser.parse_args()

    start_total = time.time()
    main(args.base_ply, args.semantic_ply, args.instance_ply, args.output_las, compress=not args.no_compress)
    print(f"\nTotal processing time: {time.time() - start_total:.2f} seconds.")

