import io
import logging

import pandas as pd
import requests
from Bio import SeqIO
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)


def _create_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504, 429)):
    """Creates a requests session with built-in retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def pdb_to_seq_uniprot(pdb_id, return_type="list", timeout=10):
    """
    Retrieve the canonical UniProt amino-acid sequence for a given PDB ID.

    Parameters
    ----------
    pdb_id : str
        PDB ID (e.g., '1a3n').
    return_type : {'list', 'pd.df'}, optional, default='list'
        Format of returned value:

          - ``'list'`` : list with one amino-acid sequence
          - ``'pd.df'`` : pandas.DataFrame with a single column ['sequence']
    timeout : int, optional, default=10
        Timeout for network requests in seconds.

    Returns
    -------
    list of str or pandas.DataFrame
        Depending on ``return_type``.

    Raises
    ------
    ValueError
        If pdb_id is invalid, mapping is not found, or parsing fails.
    requests.exceptions.RequestException
        If network requests fail after retries.
    """
    if not isinstance(pdb_id, str) or not pdb_id.strip():
        raise ValueError("pdb_id must be a non-empty string")

    pdb_id = pdb_id.lower().strip()
    session = _create_session()

    # Step 1: Get PDB to UniProt mapping
    mapping_url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    try:
        mapping_resp = session.get(mapping_url, timeout=timeout)
        mapping_resp.raise_for_status()
        mapping_data = mapping_resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch mapping for PDB ID '{pdb_id}': {e}")
        raise

    uniprot_dict = mapping_data.get(pdb_id, {}).get("UniProt", {})
    uniprot_ids = list(uniprot_dict.keys())

    if not uniprot_ids:
        raise ValueError(f"No UniProt mapping found for PDB ID '{pdb_id}'")

    if len(uniprot_ids) > 1:
        logger.warning(
            f"PDB ID '{pdb_id}' maps to multiple UniProt IDs: {uniprot_ids}. "
            f"Using the first one: {uniprot_ids[0]}"
        )

    uniprot_id = uniprot_ids[0]

    # Step 2: Get FASTA sequence from UniProt
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        fasta_resp = session.get(fasta_url, timeout=timeout)
        fasta_resp.raise_for_status()
        fasta_data = fasta_resp.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch FASTA for UniProt ID '{uniprot_id}': {e}")
        raise

    # Step 3: Parse FASTA safely
    records = list(SeqIO.parse(io.StringIO(fasta_data), "fasta"))
    if not records:
        raise ValueError(
            f"Could not parse sequence from FASTA data for UniProt ID '{uniprot_id}'"
        )

    sequence = str(records[0].seq)
    df = pd.DataFrame({"sequence": [sequence]})

    if return_type == "list":
        return df["sequence"].tolist()
    elif return_type == "pd.df":
        return df.reset_index(drop=True)
    else:
        raise ValueError("`return_type` must be either 'list' or 'pd.df'")
