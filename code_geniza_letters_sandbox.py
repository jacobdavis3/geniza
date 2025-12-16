#!/usr/bin/env python3
"""
Geniza Letter Coding System — AI Sandbox / Portkey variant

This file is a drop-in version of `code_geniza_letters.py` that routes all
model calls through the AI Sandbox Portkey gateway instead of calling the
OpenAI / Anthropic / Gemini APIs directly.

Usage (bash):
    export AI_SANDBOX_KEY="<your-sandbox-key>"
    pip install -r requirements.txt
    python code_geniza_letters_sandbox.py --document-ids <ids or path> [other args]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

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


# Goldberg's 10 Categories
CATEGORIES = [
    "Transactions",
    "Behavior of Associates",
    "Information",
    "Correspondence",
    "Travels",
    "Personal, Familial, Communal News",
    "Accounts",
    "Advice",
    "Legal Actions",
    "Government Relations",
]

# Category descriptions for prompts
CATEGORY_DESCRIPTIONS = {
    "Transactions": "Reports on actions taken, references to planned actions, orders for actions, or explanations for actions that had been delayed or not taken, purchase, collection, processing, transport, storage or sale of both merchandise and money",
    "Behavior of Associates": "Discussions of the behavior of merchants, whether the writer himself, the recipient, or third parties, merchants are assessing each other’s conduct – whether past, present, or prospective",
    "Information": "Market information, prices, reports on particular commodities – reports that are a mix of market prices and notes on current or expected demand, or local exchange rates, or the fulfillment of an order, ship movements – arrivals, expected arrivals, stowing, departures, convoying, etc. general reports on conditions: they might comment on the general state of the market, discuss the actions of important groups in the market (whether reporting on the demand of locals, the arrival of groups from elsewhere with goods for sale, or demands that would change the dynamics of the local market), tell of political events that might affect market conditions, and report on famine and plague",
    "Correspondence": "Address the current state of correspondence – whether one has received a letter since the last time one wrote, recriminations over a correspondent’s neglect, requests for intercession in managing correspondence with third parties, and attempts to find excuses for absent letters",
    "Travels": "Tales of their travels, particularly the various mishaps that befell them,  horrible weather, incompetent ship captains, pirate attacks, nights spent at odd shelters, and the like",
    "Personal, Familial, Communal News": "Communal issues, familial and personal affairs, and illness",
    "Accounts": "Geniza merchantsrequest accounts, note that they are making final accountings that will be sent, and acknowledge the receipt of other accounts",
    "Advice": "Business advice on the best ways to buy goods, the relative difficulty of various tasks, and even the balance between business and personal time, or  brief suggestions based on current market conditions",
    "Legal Actions": "Discussions of formal legal action – threatening a suit, sending powers of attorney, requesting provision of documents for an upcoming action",
    "Government Relations": "Discussions of actions of government or of any officer that directly intervene in the market or trade. It also includes any mention of a merchant’s dealings with government officers, whether it is a matter of negotiating customs or the more serious problem of appearing before the authorities after being denounced for failing to declare goods",
}


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


def create_segmentation_prompt(letter_content: str) -> str:
    """Create prompt for semantic segmentation of letter."""
    # Truncate very long content to avoid token limits
    max_content_length = 50000  # Leave room for prompt and response
    if len(letter_content) > max_content_length:
        raise Exception("Content is too long")
        # letter_content = letter_content[:max_content_length] + "\n[... content truncated ...]"

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


def create_coding_prompt(segment_text: str, mode: str) -> str:
    """Create prompt for coding a segment into categories."""
    categories_list = "\n".join(
        [f"- {cat}: {CATEGORY_DESCRIPTIONS[cat]}" for cat in CATEGORIES]
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

    return f"""You are coding a segment from a Geniza mercantile letter according to Jessica Goldberg's 10 categories.

{categories_list}

Segment to code:
{segment_text}

{instruction}

Return your response as JSON with:
- "categories": array of category names (use exact category names from the list above)
- "explanation": brief explanation of why this/these category/categories were chosen

Example format for single mode:
{{"categories": ["Transactions"], "explanation": "..."}}

Example format for multiple mode:
{{"categories": ["Transactions", "Information"], "explanation": "..."}}

Return ONLY the JSON object, no other text."""


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
    model_name: str, segment_text: str, mode: str, model_map: Dict[str, str] = None
) -> List[str]:
    """Code a segment using the specified AI model (via Portkey)."""
    prompt = create_coding_prompt(segment_text, mode)

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
        categories = parsed["categories"]
        if isinstance(categories, list):
            # Validate categories
            valid_categories = [cat for cat in categories if cat in CATEGORIES]
            return valid_categories
        elif isinstance(categories, str):
            return [categories] if categories in CATEGORIES else []

    return []


def calculate_percentages(segments: List[Dict], mode: str) -> Dict[str, float]:
    """Calculate percentage of letter for each category."""
    category_chars: Dict[str, float] = defaultdict(int)
    total_chars = 0

    for segment in segments:
        start = segment.get("start_char", 0)
        end = segment.get("end_char", start)
        segment_length = end - start
        total_chars += segment_length

        categories = segment.get("categories", [])
        if not categories:
            continue

        if mode == "single":
            # Each segment contributes its full length to one category
            if categories:
                category_chars[categories[0]] += segment_length
        else:
            # Each segment contributes equally to all its categories
            if categories:
                chars_per_category = segment_length / len(categories)
                for cat in categories:
                    category_chars[cat] += chars_per_category

    # Calculate percentages
    percentages: Dict[str, float] = {}
    if total_chars > 0:
        for category in CATEGORIES:
            percentages[category] = category_chars[category] / total_chars
    else:
        for category in CATEGORIES:
            percentages[category] = 0.0

    return percentages


def code_letter(model_name: str, letter_content: str, mode: str) -> Dict:
    """Code a complete letter using the specified model."""
    print(f"  Segmenting letter with {model_name}...")
    segments = segment_letter(model_name, letter_content)

    if not segments:
        print(f"  Warning: No segments identified by {model_name}")
        return {"codings": {cat: 0.0 for cat in CATEGORIES}, "segments": []}

    print(f"  Found {len(segments)} segments, coding each segment...")

    # Code each segment
    coded_segments: List[Dict] = []
    for i, segment in enumerate(segments):
        segment_text = segment.get("text", "")
        if not segment_text:
            continue

        print(f"    Coding segment {i+1}/{len(segments)}...")
        categories = code_segment(model_name, segment_text, mode)

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
    percentages = calculate_percentages(coded_segments, mode)

    return {"codings": percentages, "segments": coded_segments}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Code Geniza mercantile letters according to Jessica Goldberg's "
            "categories (AI Sandbox / Portkey version)"
        )
    )
    parser.add_argument(
        "--document-ids",
        required=True,
        help="Comma-separated list of document IDs or path to file with one ID per line",
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
        "--models",
        default="gpt5",
        help=(
            "Comma-separated list of logical model names to use. "
            "These are mapped to AI Sandbox deployment names."
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
        default="geniza_codings_sandbox.json",
        help="Output JSON file path (default: geniza_codings_sandbox.json)",
    )

    args = parser.parse_args()

    # Parse document IDs
    if os.path.isfile(args.document_ids):
        with open(args.document_ids, "r", encoding="utf-8") as f:
            document_ids = [
                int(line.strip()) for line in f if line.strip().isdigit()
            ]
    else:
        document_ids = [
            int(id_.strip())
            for id_ in args.document_ids.split(",")
            if id_.strip().isdigit()
        ]

    if not document_ids:
        print("Error: No valid document IDs provided")
        sys.exit(1)

    print(f"Processing {len(document_ids)} document(s): {document_ids}")

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

    requested_models = [m.strip() for m in args.models.split(",")]
    models_to_use = [m for m in requested_models if m in available_models]

    if not models_to_use:
        print("Error: No valid models specified")
        print(f"Available models: {list(available_models.keys())}")
        sys.exit(1)

    print(f"Using models: {models_to_use}")
    print(f"Model mappings: {[(m, available_models[m]) for m in models_to_use]}")
    print(f"Coding mode: {args.mode}")

    # Extract letters
    letters = extract_letters_from_csv(args.csv, document_ids)

    if not letters:
        print("Error: No letters extracted")
        sys.exit(1)

    # Code each letter with each model
    results: Dict[int, Dict] = {}
    for doc_id, letter_content in letters.items():
        print(f"\nProcessing document_id {doc_id} ({len(letter_content)} characters)...")
        results[doc_id] = {
            "total_characters": len(letter_content),
            "mode": args.mode,
            "codings": {},
            "segments": [],
        }

        # Get segments from first model (all models will code the same segments for comparison)
        print(f"\nSegmenting letter with {models_to_use[0]}...")
        segments = segment_letter(models_to_use[0], letter_content, available_models)

        if not segments:
            print(f"Warning: No segments identified. Skipping document {doc_id}")
            results[doc_id]["codings"] = {
                model: {cat: 0.0 for cat in CATEGORIES} for model in models_to_use
            }
            continue

        print(f"Found {len(segments)} segments. Coding with all models...")

        # Code segments with each model
        # Structure: segments will have categories per model
        model_results: Dict[str, Dict[str, float]] = {}
        model_segments: Dict[str, List[Dict]] = {}  # Store segments with categories per model
        base_segments_with_categories: List[Dict] = []  # From first model

        for model_name in models_to_use:
            print(f"\nCoding with {model_name}...")
            try:
                # Code each segment
                coded_segments: List[Dict] = []
                category_chars: Dict[str, float] = defaultdict(int)
                total_chars = 0

                for i, segment in enumerate(segments):
                    segment_text = segment.get("text", "")
                    if not segment_text:
                        continue

                    print(f"    Coding segment {i+1}/{len(segments)}...")
                    categories = code_segment(
                        model_name, segment_text, args.mode, available_models
                    )

                    start = segment.get("start_char", 0)
                    end = segment.get("end_char", start + len(segment_text))
                    segment_length = end - start
                    total_chars += segment_length

                    coded_segment = {
                        "text": segment_text,
                        "start_char": start,
                        "end_char": end,
                        "categories": categories,
                    }
                    coded_segments.append(coded_segment)

                    # Add category to base_segments if this is the first model (segmentation model)
                    if model_name == models_to_use[0]:
                        base_segment = {
                            "text": segment_text,
                            "start_char": start,
                            "end_char": end,
                            "category": (
                                categories[0] if categories else None
                            ),  # Primary / first category
                        }
                        base_segments_with_categories.append(base_segment)

                    # Calculate character counts
                    if categories:
                        if args.mode == "single":
                            category_chars[categories[0]] += segment_length
                        else:
                            chars_per_category = segment_length / len(categories)
                            for cat in categories:
                                category_chars[cat] += chars_per_category

                    # Rate limiting
                    time.sleep(0.5)

                # Calculate percentages
                percentages: Dict[str, float] = {}
                if total_chars > 0:
                    for category in CATEGORIES:
                        percentages[category] = category_chars[category] / total_chars
                else:
                    for category in CATEGORIES:
                        percentages[category] = 0.0

                model_results[model_name] = percentages
                model_segments[model_name] = coded_segments

            except Exception as e:
                print(f"Error coding with {model_name}: {e}")
                model_results[model_name] = {cat: 0.0 for cat in CATEGORIES}
                model_segments[model_name] = []

        # Store segments with categories per model
        results[doc_id]["codings"] = model_results
        results[doc_id]["segments"] = {
            "base_segments": base_segments_with_categories,
            "model_codings": model_segments,
        }

    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    main()
