import os
import numpy as np
import pickle
import cooler # pip install cooler
import pyfaidx # pip install pyfaidx
import torch
from tqdm import tqdm # pip install tqdm

def one_hot_encode_sequence(sequence: str, seq_length: int) -> np.ndarray:
    """
    One-hot encodes a DNA sequence.
    Pads or truncates the sequence to seq_length.

    Args:
        sequence (str): DNA sequence string (e.g., 'ATCG').
        seq_length (int): The desired fixed length for the one-hot encoded sequence.

    Returns:
        np.ndarray: One-hot encoded sequence as a NumPy array (seq_length, 4).
                    Order: A, C, G, T.
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0} # 'N' maps to A, adjust as needed
    one_hot = np.zeros((seq_length, 4), dtype=np.float32) # (Length, Channels)

    # Handle padding/truncation
    if len(sequence) < seq_length:
        sequence = sequence + 'N' * (seq_length - len(sequence)) # Pad with Ns
    elif len(sequence) > seq_length:
        sequence = sequence[:seq_length] # Truncate

    for i, char in enumerate(sequence):
        if char.upper() in mapping:
            one_hot[i, mapping[char.upper()]] = 1.0
        # For characters not in mapping, it remains zeros, which is effectively N
    return one_hot

def process_mcool_data_and_save_pickle(
    cool_file_path: str,
    fasta_file_path: str,
    output_dir: str,
    chrom_name: str,
    hic_start_bp: int, # Start BP for Hi-C matrix extraction
    hic_end_bp: int,   # End BP for Hi-C matrix extraction
    hic_resolution: int, # Resolution of the Hi-C data (e.g., 10000 for 10kb)
    input_window_size: int, # e.g., 224 for 224x224 Hi-C matrix
    sequence_length: int, # e.g., 131072 for Enformer
    apply_log_transform: bool = True, # Flag to control log10(x+1) transformation
    verbose: bool = False # Added verbose flag
) -> bool: # Returns True if successful, False otherwise
    """
    Extracts Hi-C submatrix and sequence data, and saves them to a .pkl file.
    Returns True on success, False on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    region_identifier = f"{chrom_name}:{hic_start_bp}-{hic_end_bp} ({hic_resolution/1000}kb)"
    if verbose:
        print(f"--- Processing Region: {region_identifier} from {os.path.basename(cool_file_path)} ---")

    try:
        # Construct the cooler path for the specific resolution
        cool_path_with_res = f'{cool_file_path}::/resolutions/{hic_resolution}'
        c = cooler.Cooler(cool_path_with_res)
        
        # Fetch the Hi-C submatrix for the specified genomic region
        region_matrix = c.matrix(balance=True, sparse=False).fetch(f'{chrom_name}:{hic_start_bp}-{hic_end_bp}')
        if verbose:
            print(f"  Fetched Hi-C matrix with shape: {region_matrix.shape}")

        # Handle NaNs (Not a Number) which are common in sparse Hi-C data, converting them to 0.0
        region_matrix = np.nan_to_num(region_matrix, nan=0.0)

        # Apply log transform if specified
        if apply_log_transform:
            region_matrix = np.log10(region_matrix + 1)
        
        region_matrix = region_matrix.astype(np.float32)

        # IMPORTANT: Ensure the Hi-C matrix matches the expected input_window_size
        # The fetched matrix will have dimensions (hic_end_bp - hic_start_bp) / hic_resolution
        expected_bins_from_region = (hic_end_bp - hic_start_bp) // hic_resolution
        if region_matrix.shape[0] != input_window_size or region_matrix.shape[1] != input_window_size:
            if verbose:
                print(f"  WARNING: Extracted Hi-C matrix shape {region_matrix.shape} (expected {expected_bins_from_region}x{expected_bins_from_region} bins from region definition) "
                      f"does NOT match desired model input {input_window_size}x{input_window_size}.")
                print(f"  Attempting to resize/crop. You might need to adjust hic_start_bp/hic_end_bp or resolution.")
            temp_matrix = np.zeros((input_window_size, input_window_size), dtype=np.float32)
            h, w = region_matrix.shape
            temp_matrix[:min(h, input_window_size), :min(w, input_window_size)] = \
                region_matrix[:min(h, input_window_size), :min(w, input_window_size)]
            region_matrix = temp_matrix
            if verbose:
                print(f"  Hi-C matrix resized to: {region_matrix.shape}")
        elif verbose:
            print(f"  Hi-C matrix shape {region_matrix.shape} matches desired {input_window_size}x{input_window_size}.")


    except Exception as e:
        print(f"  ERROR for {region_identifier} from {os.path.basename(cool_file_path)}: Failed to load Hi-C data. Skipping this region.")
        print(f"  Error details: {e}")
        return False # Indicate failure

    # 2. Load Sequence Data (aligned by centering)
    try:
        # Calculate the center of the Hi-C region
        hic_center_bp = hic_start_bp + ((hic_end_bp - hic_start_bp) / 2)

        # Calculate Enformer sequence window (centered around Hi-C region)
        sequence_start_bp = int(hic_center_bp - (sequence_length / 2))
        sequence_end_bp = int(hic_center_bp + (sequence_length / 2))

        genes = pyfaidx.Fasta(fasta_file_path) # Removed as_raw=True
        # Extract sequence for the specified region
        sequence_str = genes[chrom_name][sequence_start_bp:sequence_end_bp].seq
        genes.close() # Close the FASTA file

        # One-hot encode the sequence
        one_hot_seq = one_hot_encode_sequence(sequence_str, sequence_length)
        if verbose:
            print(f"  Fetched and one-hot encoded sequence of length: {len(sequence_str)} (target: {sequence_length})")

    except Exception as e:
        print(f"  ERROR for {region_identifier} from {os.path.basename(cool_file_path)}: Failed to load sequence data. Skipping this region.")
        print(f"  Error details: {e}")
        return False # Indicate failure

    # 3. Define Targets
    # For resolution enhancement (task 3), your '2d_target' would typically be
    # the higher-resolution ground truth Hi-C matrix corresponding to the input.
    # We will assume 'target_matrix' is same as 'region_matrix' for now.
    target_matrix = region_matrix 

    total_count = None # Assuming you don't calculate this explicitly for now.
    embed_target = None # Not used for resolution enhancement task.
    target_vector = None # Not used for resolution enhancement task.

    # Construct the dictionary dynamically, only including non-None values for optional keys.
    data_to_save = {
        'input': region_matrix,
        'sequence_data': one_hot_seq, # Sequence data is now mandatory for your task
    }

    if total_count is not None:
        data_to_save['input_count'] = total_count
    if target_matrix is not None: # For resolution enhancement, this is likely always present
        data_to_save['2d_target'] = target_matrix
    if embed_target is not None:
        data_to_save['embed_target'] = embed_target
    if target_vector is not None:
        data_to_save['1d_target'] = target_vector

    if verbose:
        print(f"  Data dictionary prepared with keys: {list(data_to_save.keys())}")

    base_mcool_name = os.path.basename(cool_file_path).replace('.mcool', '').replace('.cool', '')
    file_name = f"{base_mcool_name}_{chrom_name}_{hic_start_bp}_{hic_end_bp}_hicseq.pkl"
    output_path = os.path.join(output_dir, file_name)
    
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    if verbose:
        print(f"  Successfully saved .pkl file to: {output_path}")
    if verbose:
        print("-" * 70)
    return True # Indicate success

# --- Main script execution for multiple genomic regions ---
if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Customize these paths to your actual files and desired output location
    YOUR_MCOOL_FILE_PATH = "/path/to/your/hic_data.mcool" # The single .mcool file you want to process
    YOUR_FASTA_FILE = "/path/to/your/Homo_sapiens.GRCh38.dna.primary_assembly.fa" # Your genomic FASTA file
    OUTPUT_BASE_DIR = "./processed_finetune_data_multiple_regions" # Output directory for .pkl files

    # --- Model Input Parameters (as per HiCFoundation paper for resolution enhancement) ---
    MODEL_INPUT_HIC_WINDOW_SIZE = 224 # Hi-C matrix will be 224x224 bins
    HIC_DATA_RESOLUTION_BP = 8000 # Hi-C data is at 8 kb resolution (changed from 10kb as per your mcool file)
    
    # --- Sequence Parameters ---
    # Enformer's typical input sequence length (adjust if your Enformer model expects a different length)
    ENFORMER_SEQUENCE_LENGTH = 131072 # Common Enformer input length

    # Determine if Hi-C data needs log transformation
    # Keep True if your raw mcool data is not log-transformed and Finetune_Dataset expects log10(x+1)
    APPLY_LOG_TRANSFORM_TO_HIC = True 

    VERBOSE_PER_REGION = False # Set to True for detailed per-region output, False for concise progress

    # --- Prepare Output Directories ---
    train_output_dir = os.path.join(OUTPUT_BASE_DIR, "train")
    val_output_dir = os.path.join(OUTPUT_BASE_DIR, "val")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    # --- Step 1: Get Chromosome Lengths ---
    print("Loading FASTA file and getting chromosome lengths...")
    try:
        genome_fasta = pyfaidx.Fasta(YOUR_FASTA_FILE)
        chrom_lengths = {chrom.name: len(chrom) for chrom in genome_fasta}
        genome_fasta.close()
        print("FASTA file loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load FASTA file or get chromosome lengths. Error: {e}")
        exit()

    # --- Step 2: Define Chromosomes for Training and Validation/Testing ---
    # As per the HiCFoundation paper's resolution enhancement task:
    TRAIN_CHROMS = [f'chr{i}' for i in range(1, 23) if i not in [4, 5, 11, 14]] + ['chrX']
    VAL_TEST_CHROMS = ['chr4', 'chr5', 'chr11', 'chr14']

    # Filter out chromosomes that might not be in your FASTA or mcool file
    TRAIN_CHROMS = [c for c in TRAIN_CHROMS if c in chrom_lengths]
    VAL_TEST_CHROMS = [c for c in VAL_TEST_CHROMS if c in chrom_lengths]

    # --- Step 3: Generate Genomic Windows and Assign to Train/Val Sets ---
    all_regions_to_process = []
    WINDOW_SPAN_BP = MODEL_INPUT_HIC_WINDOW_SIZE * HIC_DATA_RESOLUTION_BP # e.g., 224 * 8000 = 1,792,000 bp (1.792 Mb)

    print(f"\nGenerating genomic windows (span: {WINDOW_SPAN_BP/1_000_000:.2f} Mb)...")
    for chrom_list, dataset_type in [(TRAIN_CHROMS, 'train'), (VAL_TEST_CHROMS, 'val')]:
        output_dir = train_output_dir if dataset_type == 'train' else val_output_dir
        for chrom_name in chrom_list:
            current_chrom_length = chrom_lengths.get(chrom_name)
            if not current_chrom_length:
                print(f"Skipping {chrom_name}: Length not found in FASTA.")
                continue

            for start_bp in range(0, current_chrom_length - WINDOW_SPAN_BP + 1, WINDOW_SPAN_BP):
                end_bp = start_bp + WINDOW_SPAN_BP
                
                all_regions_to_process.append({
                    'mcool_path': YOUR_MCOOL_FILE_PATH,
                    'chrom': chrom_name,
                    'hic_start_bp': start_bp,
                    'hic_end_bp': end_bp,
                    'hic_resolution': HIC_DATA_RESOLUTION_BP,
                    'output_dir': output_dir
                })
    
    print(f"Generated {len(all_regions_to_process)} total regions for processing.")
    train_region_count = len([r for r in all_regions_to_process if r['output_dir'] == train_output_dir])
    val_region_count = len([r for r in all_regions_to_process if r['output_dir'] == val_output_dir])
    print(f"  {train_region_count} training regions.")
    print(f"  {val_region_count} validation regions.")


    # --- Step 4: Process Each Generated Region with Progress Bar ---
    print("\nStarting data processing...")
    successful_regions = 0
    failed_regions = 0

    for i, region_info in tqdm(enumerate(all_regions_to_process), total=len(all_regions_to_process), desc="Processing regions"):
        success = process_mcool_data_and_save_pickle(
            cool_file_path=region_info['mcool_path'],
            fasta_file_path=YOUR_FASTA_FILE,
            output_dir=region_info['output_dir'],
            chrom_name=region_info['chrom'],
            hic_start_bp=region_info['hic_start_bp'],
            hic_end_bp=region_info['hic_end_bp'],
            hic_resolution=region_info['hic_resolution'],
            input_window_size=MODEL_INPUT_HIC_WINDOW_SIZE,
            sequence_length=ENFORMER_SEQUENCE_LENGTH,
            apply_log_transform=APPLY_LOG_TRANSFORM_TO_HIC,
            verbose=VERBOSE_PER_REGION
        )
        if success:
            successful_regions += 1
        else:
            failed_regions += 1

    print("\nAll region processing complete.")
    print(f"Summary: {successful_regions} regions processed successfully, {failed_regions} regions failed.")
    print(f"Training .pkl files saved to: {train_output_dir}")
    print(f"Validation .pkl files saved to: {val_output_dir}")
    print("\nRemember to update your --train_config and --valid_config files in your main project directory.")
    print(f"For example, if '{OUTPUT_BASE_DIR}' is your data_path, your train_config would contain 'train' and valid_config 'val'.")