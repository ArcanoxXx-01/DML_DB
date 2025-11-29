import socket
from typing import List

def resolve_peers(domain: str) -> List[str]:
    """
    Resolve the given DNS domain to a list of peer IP addresses.
    """
    try:
        # gethostbyname_ex returns (hostname, aliaslist, ipaddrlist)
        _, _, ipaddrlist = socket.gethostbyname_ex(domain)
        return ipaddrlist
    except Exception as e:
        print(f"Error resolving peers for domain {domain}: {e}")
        return []
