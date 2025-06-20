"""NASA Earthdata API adapter implementation."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import APIAdapter, APIError, AuthenticationError

logger = logging.getLogger(__name__)


class NASAEarthdataAdapter:
    """NASA Earthdata API adapter for CMR and data download.
    
    Implements the APIAdapter interface for NASA Earthdata operations,
    including CMR (Common Metadata Repository) searches and granule downloads.
    """
    
    CMR_BASE_URL = "https://cmr.earthdata.nasa.gov"
    CMR_SEARCH_URL = f"{CMR_BASE_URL}/search"
    
    def __init__(
        self,
        earthdata_token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.3
    ):
        """Initialize NASA Earthdata adapter.
        
        Args:
            earthdata_token: NASA Earthdata authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff factor for retries
        """
        self.earthdata_token = earthdata_token
        self.timeout = timeout
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set authentication header if token provided
        if earthdata_token:
            self.session.headers.update({
                'Authorization': f'Bearer {earthdata_token}'
            })
    
    def search(self, **params) -> List[Dict[str, Any]]:
        """Search for granules using CMR API.
        
        Args:
            **params: Search parameters including:
                - short_name: Collection short name (e.g., "VNP46A2")
                - version: Collection version
                - bounding_box: Comma-separated bbox string
                - temporal: Time range string
                - page_size: Results per page
                - page_num: Page number
                
        Returns:
            List of granule metadata dictionaries
            
        Raises:
            APIError: If search fails
        """
        # Build search URL
        search_url = f"{self.CMR_SEARCH_URL}/granules.json"
        
        # Ensure required parameters
        if 'short_name' not in params:
            raise APIError("short_name parameter is required for search")
        
        # Set default page size if not specified
        if 'page_size' not in params:
            params['page_size'] = 2000
            
        try:
            logger.debug(f"Searching CMR with params: {params}")
            response = self.session.get(
                search_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            granules = data.get('feed', {}).get('entry', [])
            
            logger.info(f"Found {len(granules)} granules for {params['short_name']}")
            return granules
            
        except requests.exceptions.RequestException as e:
            raise APIError(f"CMR search failed: {e}")
        except Exception as e:
            raise APIError(f"Unexpected error during search: {e}")
    
    def download(self, resource_id: str, output_path: Optional[Path] = None) -> bytes:
        """Download granule data.
        
        Args:
            resource_id: Granule URL or identifier
            output_path: Optional local path to save file
            
        Returns:
            Downloaded data as bytes
            
        Raises:
            APIError: If download fails
        """
        if not self.earthdata_token:
            raise AuthenticationError("Earthdata token required for downloads")
        
        try:
            # If resource_id is not a full URL, assume it's a granule ID
            if not resource_id.startswith('http'):
                raise APIError("Full URL required for download")
            
            logger.debug(f"Downloading from {resource_id}")
            
            response = self.session.get(
                resource_id,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            if output_path:
                # Stream to file for large downloads
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
                logger.debug(f"Downloaded to {output_path}")
                return output_path.read_bytes()
            else:
                # Download to memory
                data = response.content
                logger.debug(f"Downloaded {len(data)} bytes")
                return data
                
        except requests.exceptions.RequestException as e:
            raise APIError(f"Download failed: {e}")
        except Exception as e:
            raise APIError(f"Unexpected error during download: {e}")
    
    def authenticate(self, credentials: Dict[str, str]) -> None:
        """Update authentication credentials.
        
        Args:
            credentials: Dictionary with 'token' key
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        if 'token' not in credentials:
            raise AuthenticationError("Token required in credentials")
            
        self.earthdata_token = credentials['token']
        self.session.headers.update({
            'Authorization': f'Bearer {self.earthdata_token}'
        })
        
        # Verify token by making a simple request
        try:
            response = self.session.get(
                f"{self.CMR_BASE_URL}/collections.json",
                params={'page_size': 1},
                timeout=10
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise AuthenticationError(f"Token validation failed: {e}")
    
    def search_granules(
        self,
        collection: str,
        version: str,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for granules with specific parameters.
        
        Args:
            collection: Collection short name
            version: Collection version
            bbox: Bounding box (west, south, east, north)
            start_date: Start of temporal range
            end_date: End of temporal range
            max_results: Maximum number of results to return
            
        Returns:
            List of granule metadata
        """
        # Format temporal range
        temporal = f"{start_date.isoformat()}Z,{end_date.isoformat()}Z"
        
        # Format bounding box
        bounding_box = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        
        params = {
            'short_name': collection,
            'version': version,
            'bounding_box': bounding_box,
            'temporal': temporal,
            'page_size': min(max_results or 2000, 2000)
        }
        
        all_granules = []
        page_num = 1
        
        while True:
            params['page_num'] = page_num
            granules = self.search(**params)
            
            if not granules:
                break
                
            all_granules.extend(granules)
            
            if max_results and len(all_granules) >= max_results:
                all_granules = all_granules[:max_results]
                break
                
            if len(granules) < params['page_size']:
                break
                
            page_num += 1
            
        return all_granules
    
    def get_download_urls(self, granule: Dict[str, Any]) -> List[str]:
        """Extract download URLs from granule metadata.
        
        Args:
            granule: Granule metadata from CMR
            
        Returns:
            List of download URLs
        """
        urls = []
        
        # Check for links in the granule
        links = granule.get('links', [])
        for link in links:
            if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                href = link.get('href')
                if href:
                    urls.append(href)
        
        # Also check for online_access_urls
        online_urls = granule.get('online_access_urls', [])
        urls.extend(online_urls)
        
        return urls
    
    def wait_between_requests(self, delay: float = 0.1) -> None:
        """Add delay between requests to avoid rate limiting.
        
        Args:
            delay: Delay in seconds
        """
        time.sleep(delay)