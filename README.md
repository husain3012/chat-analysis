# Project Title

Whatsapp Chat Analyzer (WIP)
## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
    - [Prerequisites](#prerequisites)
    - [Installing](#installing)
  - [Usage ](#usage-)
    - [CLI options](#cli-options)

## About <a name = "about"></a>

Analyze your whatsapp chat using AI, locally.

### Prerequisites

What things you need to install the software and how to install them.

- Python 3.6 or higher
- Internet connection (for downloading the deep learning model first time)
- Good GPU (optional)


### Installing


- Clone this repo

- `cd` into `chat-analysis/`

- Create a virtual environment

    ```bash
    python -m venv venv
    ```
- Activate the virtual environment

    Unix
    ```bash
    source venv/bin/activate
    ```
    Windows
    ```bash
    venv\Scripts\activate
    ``` 

- Install the requirements

    ```bash
    pip install -r requirements.txt
    ```
- Test cli

    ```bash
    python cli.py --help
    ```

## Usage <a name = "usage"></a>

- Export your whatsapp chat using the `without media` option, save the `zip` file
- Run the cli

    ```bash
    python cli.py  path/to/zip/file
    ```
- The cli will save the output in json format in `data/processed/` directory
- Your chat file will be saved in `data/uploads/whatsapp_chat` directory, make sure to delete it after the analysis is done

### CLI options

`--help` : Shows help \
`--no-deep` : Disables deep learning analysis\
`--sample-size` : Selects a continuous subset from the chat randomly.\
`--batch-size` : Batch size to use in deep learning inference.

