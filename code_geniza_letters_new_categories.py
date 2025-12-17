#!/usr/bin/env python3
"""
Geniza Letter Coding System — AI Sandbox Batch Processing with LLM-Discovered Categories

This script reads document IDs from a text file, extracts letter transcriptions,
has the LLM determine 10 categories by analyzing all letters, then codes each
letter into those discovered categories.

Usage (bash):
    export AI_SANDBOX_KEY="<your-sandbox-key>"
    pip install -r requirements.txt
    python code_geniza_letters_sandbox_batch.py --input-file ib.txt [other args]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from portkey_ai import Portkey


# Load environment variables (e.g., from .env)
load_dotenv()

# Load AI Sandbox API key from environment
AI_SANDBOX_KEY = os.getenv("AI_SANDBOX_KEY")

if AI_SANDBOX_KEY is None or AI_SANDBOX_KEY.strip() == "":
    raise RuntimeError(
        "AI_SANDBOX_KEY environment variable is not set. "
        "Please export AI_SANDBOX_KEY before running this script."
    )

# Initialize a Portkey client using the AI Sandbox key
client = Portkey(api_key=AI_SANDBOX_KEY)


def read_document_ids_from_file(file_path: str) -> List[int]:
    """
    Read document IDs from a text file (one per line).

    Args:
        file_path: Path to the text file containing document IDs

    Returns:
        List of document IDs (integers)
    """
    document_ids: List[int] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    document_ids.append(int(line))
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)

    if not document_ids:
        print(f"Warning: No valid document IDs found in {file_path}")
        sys.exit(1)

    print(f"Read {len(document_ids)} document IDs from {file_path}")
    return document_ids


def extract_letters_from_csv(csv_path: str, document_ids: List[int]) -> Dict[int, str]:
    """
    Extract Judeo-Arabic content from CSV for given document IDs.

    Args:
        csv_path: Path to the CSV file
        document_ids: List of document IDs to extract

    Returns:
        Dictionary mapping document_id to full letter content (Judeo-Arabic text)
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter by document_ids
    df_filtered = df[df["document_id"].isin(document_ids)].copy()

    if df_filtered.empty:
        print("Warning: No documents found for the provided document IDs")
        return {}

    # Group by document_id and combine content
    letters: Dict[int, str] = {}
    for doc_id in document_ids:
        doc_rows = df_filtered[df_filtered["document_id"] == doc_id]

        # Extract content from rows that have content
        content_parts: List[str] = []
        for _, row in doc_rows.iterrows():
            content = str(row.get("content", "")).strip()
            if content and content != "nan":
                # Check if content contains Hebrew script (Judeo-Arabic)
                if re.search(r"[\u0590-\u05FF]", content):
                    content_parts.append(content)

        if content_parts:
            # Join content parts with newlines
            full_content = "\n".join(content_parts)
            letters[doc_id] = full_content
            print(f"Extracted {len(full_content)} characters for document_id {doc_id}")
        else:
            print(f"Warning: No Judeo-Arabic content found for document_id {doc_id}")

    return letters


def call_portkey_chat(
    model: str, prompt: str, use_json_mode: bool = False, max_retries: int = 3
) -> Optional[str]:
    """Call a model via the AI Sandbox / Portkey gateway."""
    for attempt in range(max_retries):
        try:
            # NOTE: Some AI Sandbox / Azure deployments only support the default
            # temperature. We omit it here so the provider can use its default.
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that analyzes historical "
                            "documents. Always respond with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }

            # Use JSON mode for coding (which returns objects), not segmentation (arrays)
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"Portkey API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                print(f"Error calling Portkey model {model}: {e}")
                return None


def parse_json_response(response: str) -> Optional[Union[dict, list]]:
    """Parse JSON from API response, handling markdown code blocks and both objects and arrays."""
    if not response:
        return None

    # Remove markdown code blocks if present
    response = response.strip()
    if response.startswith("```"):
        # Extract JSON from code block
        lines = response.split("\n")
        json_lines: List[str] = []
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code or (not in_code and line.strip()):
                json_lines.append(line)
        response = "\n".join(json_lines).strip()

    # Try to parse the JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        # Try to fix common issues with truncated JSON
        # If it's an array that's incomplete, try to extract complete items
        if response.strip().startswith("["):
            # Try to find the last complete object
            try:
                brace_count = 0
                in_string = False
                escape_next = False
                last_complete_pos = -1

                for i, char in enumerate(response):
                    if escape_next:
                        escape_next = False
                        continue
                    if char == "\\":
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                last_complete_pos = i

                if last_complete_pos > 0:
                    # Extract up to the last complete object and close the array
                    partial_response = response[: last_complete_pos + 1] + "\n]"
                    return json.loads(partial_response)
            except Exception:
                pass

        # If all else fails, try to extract what we can
        print(f"Warning: Failed to parse JSON response: {e}")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:500]}...")

        # Try to find and extract just the JSON array/object if it's embedded in text
        for start_char in ["[", "{"]:
            start_idx = response.find(start_char)
            if start_idx >= 0:
                # Try to find matching closing bracket
                end_char = "]" if start_char == "[" else "}"
                # Count brackets to find the end
                count = 0
                in_string = False
                escape_next = False
                for i in range(start_idx, len(response)):
                    char = response[i]
                    if escape_next:
                        escape_next = False
                        continue
                    if char == "\\":
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if not in_string:
                        if char == start_char:
                            count += 1
                        elif char == end_char:
                            count -= 1
                            if count == 0:
                                try:
                                    return json.loads(response[start_idx : i + 1])
                                except Exception:
                                    pass
                                break

        return None


def discover_categories(
    letters: Dict[int, str], model_name: str, model_map: Dict[str, str] = None
) -> Dict[str, str]:
    """
    Have the LLM analyze all letters and determine 10 categories.

    Args:
        letters: Dictionary mapping document_id to letter content
        model_name: Model name to use
        model_map: Mapping from short names to actual API model names

    Returns:
        Dictionary mapping category name to description
    """
    print(f"\n=== Phase 1: Discovering Categories ===")
    print(f"Analyzing {len(letters)} letters to determine 10 categories...")

    # Combine all letter content (with document IDs for context)
    # Limit total content to avoid token limits - use representative samples
    max_total_chars = 100000  # Reasonable limit for category discovery
    combined_content_parts: List[str] = []
    total_chars = 0

    for doc_id, content in letters.items():
        if total_chars + len(content) > max_total_chars:
            # Add a truncated version
            remaining = max_total_chars - total_chars
            if remaining > 1000:  # Only add if we have meaningful space
                combined_content_parts.append(
                    f"\n--- Document ID {doc_id} (truncated) ---\n{content[:remaining]}"
                )
            break
        combined_content_parts.append(f"\n--- Document ID {doc_id} ---\n{content}")
        total_chars += len(content)

    combined_content = "\n".join(combined_content_parts)

    prompt = f"""You are analyzing a collection of Geniza mercantile letters written in Judeo-Arabic (Arabic written in Hebrew script).

Your task is to analyze these letters and determine 10 categories that best describe the themes, topics, and content found across all the letters. These categories should capture the main types of content and themes present in the letters.

Here are the letters:

{combined_content}

Based on your analysis of these letters, determine exactly 10 categories that best describe their content. For each category, provide:
1. A clear, concise category name
2. A brief description of what content belongs in this category

Return your response as a JSON object where each key is a category name and each value is the description:

{{
  "Category Name 1": "Description of what content belongs in this category",
  "Category Name 2": "Description of what content belongs in this category",
  ...
}}

You must provide exactly 10 categories. Return ONLY the JSON object, no other text."""

    # Map short name to actual API model name if mapping provided
    api_model_name = model_map.get(model_name, model_name) if model_map else model_name

    response = call_portkey_chat(api_model_name, prompt, use_json_mode=True)

    if not response:
        raise RuntimeError("Failed to get response from LLM for category discovery")

    parsed = parse_json_response(response)
    if not parsed or not isinstance(parsed, dict):
        raise RuntimeError(
            f"Failed to parse category discovery response. Got: {type(parsed)}"
        )

    # Validate we got exactly 10 categories
    if len(parsed) != 10:
        print(
            f"Warning: Expected 10 categories, but got {len(parsed)}. "
            f"Proceeding with {len(parsed)} categories."
        )

    print(f"Discovered {len(parsed)} categories:")
    for cat_name, description in parsed.items():
        print(f"  - {cat_name}: {description[:80]}...")

    return parsed


def create_segmentation_prompt(letter_content: str) -> str:
    """Create prompt for semantic segmentation of letter."""
    # Truncate very long content to avoid token limits
    max_content_length = 50000  # Leave room for prompt and response
    if len(letter_content) > max_content_length:
        raise Exception("Content is too long")

    return f"""You are analyzing a Geniza mercantile letter written in Judeo-Arabic (Arabic written in Hebrew script).

Your task is to identify semantic segments in the letter. A segment can be as small as a few words - any portion of text that discusses a distinct topic or theme. Segments can be:
- A few words
- A phrase
- A sentence
- Multiple sentences
- A paragraph
- Any meaningful unit that discusses a single topic

IMPORTANT: Identify as many segments as needed to capture all topic changes in the letter. There is no limit on the number of segments. Be thorough and identify all distinct topics, even if they are very short.

Letter content:
{letter_content}

Identify ALL semantic segments in this letter. For each segment, provide:
1. The exact text of the segment (copy it exactly as it appears)
2. The starting character position (0-indexed, relative to the full letter)
3. The ending character position (0-indexed, exclusive, relative to the full letter)

Return your response as a JSON array of objects, where each object has:
- "text": the segment text (can be as short as a few words)
- "start_char": starting character position
- "end_char": ending character position

Example format:
[
  {{"text": "first segment text", "start_char": 0, "end_char": 150}},
  {{"text": "second segment", "start_char": 150, "end_char": 200}},
  {{"text": "very short segment", "start_char": 200, "end_char": 220}}
]

Return ONLY the JSON array, no other text. Ensure all JSON strings are properly escaped. Identify as many segments as you find - there is no upper limit."""


def create_coding_prompt_with_discovered_categories(
    segment_text: str, categories: Dict[str, str], mode: str
) -> str:
    """Create prompt for coding a segment into discovered categories."""
    categories_list = "\n".join(
        [f"- {cat}: {desc}" for cat, desc in categories.items()]
    )

    if mode == "single":
        instruction = (
            "Assign this segment to ONE primary category that best describes its main topic."
        )
    else:
        instruction = (
            "Assign this segment to ALL categories that are relevant. "
            "A segment can belong to multiple categories if it discusses multiple topics."
        )

    return f"""You are coding a segment from a Geniza mercantile letter according to the following categories:

{categories_list}

Segment to code:
{segment_text}

{instruction}

Return your response as JSON with:
- "categories": array of category names (use exact category names from the list above)
- "explanation": brief explanation of why this/these category/categories were chosen

Example format for single mode:
{{"categories": ["Category Name 1"], "explanation": "..."}}

Example format for multiple mode:
{{"categories": ["Category Name 1", "Category Name 2"], "explanation": "..."}}

Return ONLY the JSON object, no other text."""


def segment_letter(
    model_name: str, letter_content: str, model_map: Dict[str, str] = None
) -> List[Dict]:
    """Segment a letter using the specified AI model (via Portkey)."""
    prompt = create_segmentation_prompt(letter_content)

    # Map short name to actual API model name if mapping provided
    api_model_name = model_map.get(model_name, model_name) if model_map else model_name

    # Segmentation returns arrays, so no JSON mode
    response = call_portkey_chat(api_model_name, prompt, use_json_mode=False)

    if not response:
        return []

    # Parse response
    parsed = parse_json_response(response)
    if not parsed:
        return []

    # Handle both array and object responses
    if isinstance(parsed, list):
        return parsed
    elif isinstance(parsed, dict) and "segments" in parsed:
        return parsed["segments"]
    else:
        print(f"Warning: Unexpected response format from {model_name}")
        return []


def code_segment(
    model_name: str,
    segment_text: str,
    categories: Dict[str, str],
    mode: str,
    model_map: Dict[str, str] = None,
) -> List[str]:
    """Code a segment using the specified AI model (via Portkey) with discovered categories."""
    prompt = create_coding_prompt_with_discovered_categories(
        segment_text, categories, mode
    )

    # Map short name to actual API model name if mapping provided
    api_model_name = model_map.get(model_name, model_name) if model_map else model_name

    # Coding returns objects, so use JSON mode
    response = call_portkey_chat(api_model_name, prompt, use_json_mode=True)

    if not response:
        return []

    # Parse response
    parsed = parse_json_response(response)
    if not parsed:
        return []

    # Extract categories
    if isinstance(parsed, dict) and "categories" in parsed:
        response_categories = parsed["categories"]
        if isinstance(response_categories, list):
            # Validate categories against discovered categories
            valid_categories = [
                cat for cat in response_categories if cat in categories
            ]
            return valid_categories
        elif isinstance(response_categories, str):
            return [response_categories] if response_categories in categories else []

    return []


def calculate_percentages(
    segments: List[Dict], categories: Dict[str, str], mode: str
) -> Dict[str, float]:
    """Calculate percentage of letter for each category."""
    category_chars: Dict[str, float] = defaultdict(int)
    total_chars = 0

    for segment in segments:
        start = segment.get("start_char", 0)
        end = segment.get("end_char", start)
        segment_length = end - start
        total_chars += segment_length

        segment_categories = segment.get("categories", [])
        if not segment_categories:
            continue

        if mode == "single":
            # Each segment contributes its full length to one category
            if segment_categories:
                cat = segment_categories[0]
                if cat in categories:
                    category_chars[cat] += segment_length
        else:
            # Each segment contributes equally to all its categories
            if segment_categories:
                chars_per_category = segment_length / len(segment_categories)
                for cat in segment_categories:
                    if cat in categories:
                        category_chars[cat] += chars_per_category

    # Calculate percentages
    percentages: Dict[str, float] = {}
    if total_chars > 0:
        for category in categories.keys():
            percentages[category] = category_chars[category] / total_chars
    else:
        for category in categories.keys():
            percentages[category] = 0.0

    return percentages


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Code Geniza mercantile letters with LLM-discovered categories "
            "(AI Sandbox / Portkey version)"
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to text file with document IDs (one per line)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multiple"],
        default="single",
        help=(
            "Coding mode: 'single' (one category per segment) or "
            "'multiple' (multiple categories allowed)"
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt5",
        help=(
            "Logical model name to use. These are mapped to AI Sandbox deployment names. "
            "(default: gpt5)"
        ),
    )
    parser.add_argument(
        "--csv",
        default="princetongenizalab_pgp-metadata__data_footnotes-csv__11_29_2025.csv",
        help=(
            "Path to CSV file "
            "(default: princetongenizalab_pgp-metadata__data_footnotes-csv__11_29_2025.csv)"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: auto-generated from input filename)",
    )

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        input_basename = os.path.splitext(os.path.basename(args.input_file))[0]
        args.output = f"{input_basename}_codings_sandbox.json"
        print(f"Output file not specified, using: {args.output}")

    # Map short names to actual AI Sandbox model deployment names
    available_models: Dict[str, str] = {
        # GPT-family models via Azure/OpenAI through Portkey
        "gpt5": "gpt-5",
        "o3_mini": "o3-mini",
        "gpt4o_mini": "gpt-4o-mini",
        "gpt4o": "gpt-4o",
        "gpt4_turbo": "gpt-4-turbo",
        "gpt35_16k": "gpt-35-turbo-16k",
        # Llama via Meta
        "llama3_70b": "Llama-3.3-70B-Instruct",
        "llama3_8b": "Meta-Llama-3-1-8B-Instruct",
        # Mistral
        "mistral_small": "mistral-small-2503",
        "mistral_large": "Mistral-Large-2411",
        # Gemini via Google
        "gemini3": "gemini-3-pro-preview",
    }

    if args.model not in available_models:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(available_models.keys())}")
        sys.exit(1)

    api_model_name = available_models[args.model]
    print(f"Using model: {args.model} -> {api_model_name}")
    print(f"Coding mode: {args.mode}")

    # Phase 1: Read document IDs and extract letters
    print(f"\n=== Reading Document IDs ===")
    document_ids = read_document_ids_from_file(args.input_file)

    print(f"\n=== Extracting Letters ===")
    letters = extract_letters_from_csv(args.csv, document_ids)

    if not letters:
        print("Error: No letters extracted")
        sys.exit(1)

    # Phase 2: Discover categories
    try:
        discovered_categories = discover_categories(
            letters, args.model, available_models
        )
    except Exception as e:
        print(f"Error during category discovery: {e}")
        sys.exit(1)

    # Phase 3: Code each letter
    print(f"\n=== Phase 2: Coding Letters ===")
    print(f"Coding {len(letters)} letters into discovered categories...")

    results: Dict[int, Dict] = {}
    for doc_id, letter_content in letters.items():
        print(f"\nProcessing document_id {doc_id} ({len(letter_content)} characters)...")
        try:
            # Segment the letter
            print(f"  Segmenting letter...")
            segments = segment_letter(args.model, letter_content, available_models)

            if not segments:
                print(f"  Warning: No segments identified for document {doc_id}")
                results[doc_id] = {
                    "total_characters": len(letter_content),
                    "mode": args.mode,
                    "model": args.model,
                    "codings": {cat: 0.0 for cat in discovered_categories.keys()},
                    "segments": [],
                }
                continue

            print(f"  Found {len(segments)} segments, coding each segment...")

            # Code each segment
            coded_segments: List[Dict] = []
            for i, segment in enumerate(segments):
                segment_text = segment.get("text", "")
                if not segment_text:
                    continue

                print(f"    Coding segment {i+1}/{len(segments)}...")
                categories = code_segment(
                    args.model, segment_text, discovered_categories, args.mode, available_models
                )

                coded_segment = {
                    "text": segment_text,
                    "start_char": segment.get("start_char", 0),
                    "end_char": segment.get("end_char", len(segment_text)),
                    "categories": categories,
                }
                coded_segments.append(coded_segment)

                # Rate limiting
                time.sleep(0.5)

            # Calculate percentages
            percentages = calculate_percentages(
                coded_segments, discovered_categories, args.mode
            )

            results[doc_id] = {
                "total_characters": len(letter_content),
                "mode": args.mode,
                "model": args.model,
                "codings": percentages,
                "segments": coded_segments,
            }

        except Exception as e:
            print(f"  Error processing document {doc_id}: {e}")
            results[doc_id] = {
                "total_characters": len(letter_content),
                "mode": args.mode,
                "model": args.model,
                "codings": {cat: 0.0 for cat in discovered_categories.keys()},
                "segments": [],
                "error": str(e),
            }

    # Save results
    output_data = {
        "discovered_categories": discovered_categories,
        "letters": results,
    }

    print(f"\n=== Saving Results ===")
    print(f"Saving to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()
