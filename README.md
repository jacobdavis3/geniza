# Geniza Letter Coding System

This system codes Geniza mercantile letters according to Jessica Goldberg's 10 categories using multiple AI models. It segments letters semantically, codes each segment, and calculates the percentage of each letter devoted to each category.

## Goldberg's 10 Categories

1. **Transactions** (48.5%) - Commercial transactions, buying, selling, orders, shipments, payments, debts, credits
2. **Behavior of Associates** (18.0%) - Comments on behavior, reliability, trustworthiness, or conduct of business associates, partners, or agents
3. **Information** (11.9%) - Market information, prices, news about goods, trade conditions, general business intelligence
4. **Correspondence** (7.4%) - Matters related to letter writing itself, requests for letters, acknowledgments of receipt, postal matters
5. **Travels** (5.2%) - Travel plans, journeys, routes, arrival/departure information, travel conditions
6. **Personal, Familial, Communal News** (4.8%) - Personal news, family matters, community events, social information unrelated to business
7. **Accounts** (3.0%) - Financial accounts, bookkeeping, detailed financial records, account settlements
8. **Advice** (2.2%) - Requests for or giving of advice, counsel, recommendations
9. **Legal Actions** (1.4%) - Legal matters, disputes, court cases, legal procedures, contracts
10. **Government Relations** (1.1%) - Interactions with government officials, taxes, regulations, official matters

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys by copying `.env.example` to `.env` and adding your API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

You'll need API keys for:
- OpenAI (for GPT-4 and GPT-3.5)
- Anthropic (for Claude 3 and Claude 3.5)
- Google (for Gemini)

## Usage

### Basic Usage

Code letters by providing document IDs:

```bash
python code_geniza_letters.py --document-ids 39481,5557,35834
```

### Using a File of Document IDs

Create a text file with one document ID per line:

```bash
python code_geniza_letters.py --document-ids document_ids.txt
```

### Options

- `--document-ids`: Comma-separated list of document IDs or path to file with one ID per line (required)
- `--mode`: Coding mode - `single` (one category per segment) or `multiple` (multiple categories allowed). Default: `single`
- `--models`: Comma-separated list of models to use. Options: `gpt4`, `gpt35`, `claude3`, `claude35`, `gemini`. Default: all models
- `--csv`: Path to CSV file. Default: `princetongenizalab_pgp-metadata__data_footnotes-csv__11_29_2025.csv`
- `--output`: Output JSON file path. Default: `geniza_codings.json`

### Examples

Code with single category mode (default):
```bash
python code_geniza_letters.py --document-ids 39481 --mode single
```

Code with multiple categories allowed:
```bash
python code_geniza_letters.py --document-ids 39481 --mode multiple
```

Use only specific models:
```bash
python code_geniza_letters.py --document-ids 39481 --models gpt4,claude35
```

## Output Format

The output is a JSON file with the following structure:

```json
{
  "document_id": {
    "total_characters": 5000,
    "mode": "single",
    "codings": {
      "gpt4": {
        "Transactions": 0.485,
        "Behavior of Associates": 0.180,
        "Information": 0.119,
        ...
      },
      "gpt35": {...},
      "claude3": {...},
      "claude35": {...},
      "gemini": {...}
    },
    "segments": {
      "base_segments": [
        {
          "text": "segment text...",
          "start_char": 0,
          "end_char": 150
        }
      ],
      "model_codings": {
        "gpt4": [
          {
            "text": "segment text...",
            "start_char": 0,
            "end_char": 150,
            "categories": ["Transactions"]
          }
        ],
        "gpt35": [...],
        "claude3": [...],
        "claude35": [...],
        "gemini": [...]
      }
    }
  }
}
```

The `codings` object contains percentages (0.0 to 1.0) for each category as determined by each model. The `segments` object contains:
- `base_segments`: The original semantic segments identified (without categories)
- `model_codings`: For each model, an array of segments with their category assignments. Each segment includes the text, character positions, and the categories assigned by that model.

## How It Works

1. **Extraction**: The script reads the CSV file and extracts Judeo-Arabic content (text written in Hebrew script) for the specified document IDs.

2. **Segmentation**: The first model identifies semantic segments in the letter. Segments can be as small as a few words - any portion of text that discusses a distinct topic or theme. There is no limit on the number of segments - the model identifies as many as needed to capture all topic changes.

3. **Coding**: Each segment is coded by all models according to Goldberg's categories. In "single" mode, each segment gets one primary category. In "multiple" mode, segments can belong to multiple categories. Each model's category assignments for each segment are stored in the output.

4. **Calculation**: Character counts are calculated for each category per model, and percentages are computed based on the total letter length.

5. **Comparison**: All models code the same segments, allowing for comparison of coding consistency across different AI models. The output includes category assignments for each segment from each model.

## Notes

- The script handles rate limiting and retries for API calls
- Progress is logged to the console
- The script only processes content that contains Hebrew script (Judeo-Arabic)
- Multi-row content entries in the CSV are automatically combined

