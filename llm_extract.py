import os
import requests
import json
import csv
import base64
from PIL import Image
import io
import time
import re
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = {
    "claude": "https://api.anthropic.com/v1/messages"
}


def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_claude(image_path, prompt):
    """Send an image to Claude API and get a structured response."""
    try:
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 20000,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Display token usage
            input_tokens = response_data.get('usage', {}).get('input_tokens', 0)
            output_tokens = response_data.get('usage', {}).get('output_tokens', 0)
            logging.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
            
            return response_data
        else:
            logging.error(f"API request failed with status code {response.status_code}")
            logging.error(f"Response: {response.text}")
            return None
            
    except Exception as e:
        logging.error(f"Error in Claude API request: {e}")
        return None

def extract_json_from_content(content):
    """Extract JSON from Claude's response content, trying multiple approaches."""
    # Try to find JSON within markdown code blocks

    content = content.replace('""sublocation_2"', '"sublocation_2"')

    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except:
            pass
    
    # Try to find a JSON object directly using regex pattern matching
    json_match = re.search(r'(\{[\s\S]*\})', content)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except:
            pass
    
    # Try to find the first occurrence of '{' and the last occurrence of '}'
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_str = content[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except:
            pass
    
    # Try cleaning up the content and loading as JSON directly
    try:
        return json.loads(content)
    except:
        pass
    
    # If all else fails, try to extract just the part that looks like JSON
    try:
        # Remove any text before the first '{'
        if '{' in content:
            content = content[content.find('{'):]
        
        # Remove any text after the last '}'
        if '}' in content:
            content = content[:content.rfind('}')+1]
            
        return json.loads(content)
    except:
        return None

def extract_table_data(image_path):
    """Extract data from a valuation table image."""
    extraction_prompt = """
    Extract data from this historical valuation table image. 
        
    Please identify:
    1. The Parish name(s) (shown at the top of the page or within the page)
    2. All entries in the table with the following fields, using the abbreviations specified:
       - mr (Map reference in first column, including all letters and numbers, exactly as written)
       - townland (the main place name in CAPITAL LETTERS, without any "continued" suffixes)
       - os (just the number that appears in parentheses after the townland, e.g., extract "7" from "(Ord. S. 7.)")
       - sublocation_1 (first level nested location hierarchy under the main townland, like "TOWN OF X")
       - sublocation_2 (second level nested location hierarchy, like specific area names within a town)
       - occupier (from the "Townlands and Occupiers" column, including any occupational notes in parentheses)
       - lessor (from the "Immediate Lessor" column)
       - desc (Description of Tenement, using these abbreviations: "H" for house, "O" for offices, "L" for land, 
              "B" for bog, "G" for garden. Write out other terms in full. Example: "H,O,&L" instead of "House,offices,and land")
       - area (combine A. R. P. values with spaces between them)
       - land_val (land valuation in £ s. d. format)
       - building_val (building valuation in £ s. d. format)
       - total_val (total valuation in £ s. d. format)
       - n_shared (number of occupiers sharing this property)
       - is_total (use 1 for total/summary rows, 0 for normal entries)
       - is_exemption (use 1 for exemption entries, 0 for normal entries)

    Return a JSON object with this structure:
    {
        "parishes": [
            {
                "parish": "PARISH NAME",
                "entries": [
                    {
                        "mr": "reference number and letter if present",
                        "townland": "TOWNLAND NAME",
                        "os": "just the number from parentheses",
                        "sublocation_1": "first level nested location",
                        "sublocation_2": "second level nested location",
                        "occupier": "occupier name",
                        "lessor": "lessor name",
                        "desc": "description using abbreviations (H,O,L,B,G)",
                        "area": "area measurement (A R P format)",
                        "land_val": "land valuation (£ s. d.)",
                        "building_val": "building valuation (£ s. d.)",
                        "total_val": "total valuation (£ s. d.)",
                        "n_shared": number of occupiers sharing this property,
                        "is_total": 1 or 0,
                        "is_exemption": 1 or 0
                    },
                    ...
                ]
            },
            ...
        ]
    }

    **Important notes:**
    - Only include the "parish" field for the first entry of each parish section
    - For a sequence of entries in the same townland, only include the "townland" field for the first entry in that townland
    - Similarly, only include "sublocation_1" and "sublocation_2" for the first entry in each distinct location
    - Create separate entries for EACH unique combination of information
    - Extract just the number from parentheses after the townland name for the "os" field (e.g., "7" from "Ord. S. 7.")
    - Use abbreviations for description field: "H" for house, "O" for offices, "L" for land, "B" for bog, "G" for garden

    - Carefully extract the "occupier" and "lessor" columns to ensure these exactly reflect the names in the images
    - Include any notes in parentheses with the corresponding names

    - When a row contains multiple descriptions or valuations, create a separate row for EACH description
    - For example, if an occupier has "House,offices,and gar." and "Land" in the same row with different areas and valuations, create two separate rows, repeating the occupier and lessor names
    - Each unique description must have its own row with its corresponding area and valuation details
    - Repeat the occupier name and lessor name for each row as needed to represent all information fully    
    - For entries with braces connecting multiple occupiers to one property in the "total_val" column, create separate entries but note how many share the property in "n_shared"

    - Include all letters and numbers in map references exactly as shown (e.g., "5 a", "1 A", "21 C b")
    - Remove suffixes like "—continued" or "—contd" from townland names

    - Ensure that the rows where the "description" field includes the word "total" are fully extracted. For these rows, the "occupier" and "lessor" fields will typically be empty.
    - For blank or missing values (including entries with just dashes), use an empty string
    - Use 1 for true and 0 for false in boolean fields

    **Numerical values:**
    - Very carefully extract all numerical values in the "area", "land_val", "building_val" and "total_val" columns
    - Carefully distinguish similar-looking numbers (3/8, 3/5, 8/0, 1/7, 4/6, 0/9, 2/9), noting that these are scans of old documents
    - The number "3" has a distinct curved shape with two semi-circular elements. Do not confuse it with "5" which has a horizontal line at the top and a sharp angle. 
    - The number "8" has two loops, one above the other. Do not confuse it with "0" which is a circle and do not confuse it with "3" which has two semi-circular elements.
    - The number "5" has a straight horizontal line at the top and a sharp angle. Do not confuse it with "6," which has a continuous, rounded loop that fully encloses the bottom.
    - The number "4" has a straight vertical line and a connecting horizontal stroke, forming an angular shape. Do not confuse it with "6," which has a smooth, rounded loop that encloses its lower section.
    - The number "6" has a curved, open loop at the bottom that connects to a rounded upper section. Do not confuse it with "0," which is a fully enclosed oval shape with no openings or marks.
    - Check each monetary value twice before finalizing.
    - The number "8" has two loops stacked on top of each other, forming a figure-eight shape. The number "9" has a single loop on top with a straight line extending downwards, resembling a partially completed figure-eight.
    - The number "3" has two semi-circular shapes stacked on top of each other, but they are not fully enclosed. The number "8" has two fully enclosed loops, one above the other, creating a continuous shape.
    - The number "9" has a single loop with a straight line extending down from it, resembling a partially completed circle. The number "0" is a complete circle with no extra lines extending from it.

    Return only the JSON object and no other text.
    """
    
    response = query_claude(image_path, extraction_prompt)
    
    if not response:
        return None
    
    # Extract the JSON content from the response
    try:
        content = response['content'][0]['text']
        
        # Try various approaches to extract the JSON
        data = extract_json_from_content(content)
        
        if data:
            return data
        else:
            logging.error("Couldn't extract valid JSON from the response")
            logging.error(f"Response content (first 200 chars): {content[:200]}...")
            
            # Save the full response to a file for debugging
            debug_file = f"{os.path.basename(image_path)}_response.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.error(f"Full response saved to {debug_file}")
            
            return None
    
    except Exception as e:
        logging.error(f"Error processing Claude response: {e}")
        logging.error(f"Response content: {response['content'][0]['text'] if response and 'content' in response else 'No content'}")
        return None

def save_to_csv(data, output_file):
    """Save extracted data to a CSV file."""
    if not data or 'parishes' not in data or not data['parishes']:
        logging.error("No valid data to save to CSV")
        return False
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['parish', 'mr', 'townland', 'os', 'sublocation_1', 'sublocation_2', 
                          'occupier', 'lessor', 'desc', 'area', 'land_val', 
                          'building_val', 'total_val', 'n_shared', 'is_total', 'is_exemption']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for parish_data in data['parishes']:
                parish = parish_data.get('parish', '')
                
                # Track townlands and sublocations to only include them in the first entry
                current_townland = None
                current_sublocation_1 = None
                current_sublocation_2 = None
                
                for entry in parish_data.get('entries', []):
                    # Get the townland name, handling both the new field name and backward compatibility
                    raw_townland = entry.get('townland', '')
                    
                    # Clean the townland name if the function exists, otherwise use it as-is
                    townland = raw_townland
                    if 'clean_townland_name' in globals():
                        townland = clean_townland_name(raw_townland)
                    
                    # Only include townland if it's different from the previous one
                    output_townland = ''
                    if townland != current_townland:
                        output_townland = townland
                        current_townland = townland
                        # Reset sublocations when townland changes
                        current_sublocation_1 = None
                        current_sublocation_2 = None
                    
                    # Only include sublocation_1 if it's different from the previous one
                    sublocation_1 = entry.get('sublocation_1', '')
                    output_sublocation_1 = ''
                    if sublocation_1 != current_sublocation_1:
                        output_sublocation_1 = sublocation_1
                        current_sublocation_1 = sublocation_1
                        # Reset sublocation_2 when sublocation_1 changes
                        current_sublocation_2 = None
                    
                    # Only include sublocation_2 if it's different from the previous one
                    sublocation_2 = entry.get('sublocation_2', '')
                    output_sublocation_2 = ''
                    if sublocation_2 != current_sublocation_2:
                        output_sublocation_2 = sublocation_2
                        current_sublocation_2 = sublocation_2
                    
                    # Get the OS number (either from the dedicated field or extract it)
                    os_number = entry.get('os', '')
                    if not os_number and 'extract_os_number' in globals() and sublocation_1:
                        os_number = extract_os_number(sublocation_1)
                    
                    # Handle both old and new field names for compatibility
                    mr = entry.get('mr', entry.get('map_reference', ''))
                    desc = entry.get('desc', entry.get('description', ''))
                    land_val = entry.get('land_val', entry.get('land_valuation', ''))
                    building_val = entry.get('building_val', entry.get('building_valuation', ''))
                    total_val = entry.get('total_val', entry.get('total_valuation', ''))
                    
                    row = {
                        'parish': parish,
                        'mr': mr,
                        'townland': output_townland,
                        'os': os_number,
                        'sublocation_1': output_sublocation_1,
                        'sublocation_2': output_sublocation_2,
                        'occupier': entry.get('occupier', ''),
                        'lessor': entry.get('lessor', ''),
                        'desc': desc,
                        'area': entry.get('area', ''),
                        'land_val': land_val,
                        'building_val': building_val,
                        'total_val': total_val,
                        'n_shared': entry.get('n_shared', 1),  # Default to 1 if not specified
                        'is_total': entry.get('is_total', 0),  # Default to 0 if not specified
                        'is_exemption': entry.get('is_exemption', 0)  # Default to 0 if not specified
                    }
                    writer.writerow(row)
                
        logging.info(f"Data successfully saved to {output_file}")
        return True
    
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
        return False

def process_valuation_image(image_path, output_folder):
    """Process a specific valuation image file and save results to CSV."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    logging.info(f"Processing image: {image_path}")
    
    # For debugging: try to directly use the provided JSON
    json_file = image_path + ".json"
    if os.path.exists(json_file):
        logging.info(f"Found JSON file {json_file}, using it directly")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Create output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_file = os.path.join(output_folder, f"{base_name}.csv")
            
            # Save to CSV
            save_to_csv(data, output_file)
            return output_file
        except Exception as e:
            logging.error(f"Error using provided JSON file: {e}")
    
    # Extract data
    data = extract_table_data(image_path)
    
    if data:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = os.path.join(output_folder, f"{base_name}.csv")
        
        # Save to CSV
        save_to_csv(data, output_file)
        return output_file
    else:
        logging.error(f"Failed to extract data from {image_path}")
        return None

def process_batch_from_csv():
    """
    Process a batch of images based on a CSV file containing image details.
    - Reads "claude_to_extract.csv" with columns "folder" and "target_filename"
    - Forms input path as "Nanonets/analysis/[folder]/[target_filename]"
    - Ensures [folder] has 3 characters with leading zeros (e.g., "001" for folder=1)
    - Saves output to "Nanonets/claude_output/[target_filename].csv" (replacing .jpg with .csv)
    - Skips already processed images
    """
    input_csv = "Nanonets/claude_to_extract.csv"
    output_folder = "Nanonets/claude_output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of already processed files
    processed_files = set()
    for file in os.listdir(output_folder):
        if file.endswith('.csv'):
            processed_files.add(file)
    
    try:
        # Read the CSV file
        batch_data = pd.read_csv(input_csv)
        
        #batch_data = batch_data[batch_data['folder'] == 182] # FIXME REMOVE THIS - LIMITED EXTRACTIONS
        batch_data = batch_data[batch_data['folder'].between(298,299)] 

        logging.info(f"Found {len(batch_data)} images to process in {input_csv}")
        
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        for idx, row in batch_data.iterrows():
            folder = str(row['folder']).zfill(3)  # Ensure folder has 3 characters with leading zeros
            target_filename = row['target_filename']
            
            # Form the input and output paths
            input_path = f"Nanonets/analysis/{folder}/{target_filename}"
            output_filename = os.path.splitext(target_filename)[0] + '.csv'
            output_path = os.path.join(output_folder, output_filename)
            
            # Check if already processed
            if os.path.exists(output_path) or output_filename in processed_files:
                logging.info(f"Skipping already processed file: {target_filename}")
                skipped_count += 1
                continue
            
            # Process the image
            logging.info(f"Processing image {idx+1}/{len(batch_data)}: {input_path}")
            
            if os.path.exists(input_path):
                result = process_valuation_image(input_path, output_folder)
                
                if result:
                    logging.info(f"Successfully processed: {target_filename}")
                    processed_count += 1
                else:
                    logging.error(f"Failed to process: {target_filename}")
                    failed_count += 1
            else:
                logging.error(f"Input file does not exist: {input_path}")
                failed_count += 1
            
            # Add a delay to avoid API rate limits
            time.sleep(1)
        
        logging.info(f"Batch processing complete: {processed_count} processed, {skipped_count} skipped, {failed_count} failed")
        return processed_count, skipped_count, failed_count
        
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        return 0, 0, 0

process_batch_from_csv()