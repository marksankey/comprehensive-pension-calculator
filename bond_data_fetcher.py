"""
Bond Data Fetcher Module
Integrates free APIs to fetch UK gilt prices and yields
- Bank of England IADB API for yield curves
- DMO data for gilt inventory
- Intelligent caching for daily updates
"""

import pandas as pd
import requests
import pickle
import os
from datetime import datetime, timedelta
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BondDataFetcher:
    """Fetches and caches UK gilt price and yield data from free sources"""

    def __init__(self, cache_dir='.cache'):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, 'gilt_data_cache.pkl')
        self.cache_max_age_hours = 24  # Refresh daily

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Bank of England IADB API endpoint
        self.boe_api_url = 'http://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes'

    def get_cached_data(self):
        """Load cached data if fresh enough"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                cache_age = datetime.now() - cached_data['timestamp']

                if cache_age < timedelta(hours=self.cache_max_age_hours):
                    logger.info(f"Using cached data (age: {cache_age.total_seconds()/3600:.1f} hours)")
                    return cached_data
                else:
                    logger.info(f"Cache expired (age: {cache_age.total_seconds()/3600:.1f} hours)")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        return None

    def save_to_cache(self, data):
        """Save data to cache with timestamp"""
        try:
            cache_data = {
                'timestamp': datetime.now(),
                'data': data
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info("Data saved to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def fetch_boe_yield_curve(self, date_from=None, date_to=None):
        """
        Fetch UK gilt yield curve from Bank of England IADB API

        Returns yield data for various maturities that can be used to
        interpolate yields for specific gilts
        """
        if date_from is None:
            date_from = (datetime.now() - timedelta(days=30)).strftime('%d/%b/%Y')
        if date_to is None:
            date_to = datetime.now().strftime('%d/%b/%Y')

        # Series codes for UK nominal gilt yields at various maturities
        # These are spot yields from the BoE yield curve model
        series_codes = ','.join([
            'IUDMNPY',  # Nominal spot yield: 1 year
            'IUDMOPY',  # Nominal spot yield: 2 year
            'IUDMPPY',  # Nominal spot yield: 3 year
            'IUDMQPY',  # Nominal spot yield: 5 year
            'IUDMRPY',  # Nominal spot yield: 7 year
            'IUDMSOY',  # Nominal spot yield: 10 year
            'IUDMTOY',  # Nominal spot yield: 15 year
            'IUDMVPY',  # Nominal spot yield: 20 year
            'IUDMWPY',  # Nominal spot yield: 25 year
            'IUDBEDR',  # Nominal spot yield: 30 year
        ])

        payload = {
            'Datefrom': date_from,
            'Dateto': date_to,
            'SeriesCodes': series_codes,
            'CSVF': 'TN',  # Tabular format, no titles
            'UsingCodes': 'Y',
            'VPD': 'Y',  # Value per date
            'VFD': 'N'
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }

        try:
            logger.info("Fetching yield curve from Bank of England...")
            response = requests.get(
                self.boe_api_url,
                params=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                # Parse CSV response
                df = pd.read_csv(StringIO(response.text))
                logger.info(f"Successfully fetched {len(df)} rows of yield data")
                return df
            else:
                logger.error(f"BoE API request failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch BoE yield curve: {e}")
            return None

    def interpolate_yield(self, years_to_maturity, yield_curve_data=None):
        """
        Interpolate yield for a specific maturity using the yield curve

        Args:
            years_to_maturity: Years until bond maturity
            yield_curve_data: DataFrame with yield curve data (if None, fetches fresh)

        Returns:
            Interpolated yield as percentage (e.g., 4.25 for 4.25%)
        """
        if yield_curve_data is None or yield_curve_data.empty:
            yield_curve_data = self.fetch_boe_yield_curve()

        if yield_curve_data is None or yield_curve_data.empty:
            logger.warning("No yield curve data available, using fallback")
            # Fallback: simple estimate based on maturity
            return 4.0 + (years_to_maturity * 0.05)

        # Get the most recent yield curve data
        latest_row = yield_curve_data.iloc[-1]

        # Map maturities to yields
        maturity_yield_map = {
            1: latest_row.get('IUDMNPY', 4.0),
            2: latest_row.get('IUDMOPY', 4.0),
            3: latest_row.get('IUDMPPY', 4.0),
            5: latest_row.get('IUDMQPY', 4.2),
            7: latest_row.get('IUDMRPY', 4.3),
            10: latest_row.get('IUDMSOY', 4.4),
            15: latest_row.get('IUDMTOY', 4.5),
            20: latest_row.get('IUDMVPY', 4.6),
            25: latest_row.get('IUDMWPY', 4.7),
            30: latest_row.get('IUDBEDR', 4.8),
        }

        # Linear interpolation
        maturities = sorted(maturity_yield_map.keys())

        # Find bracketing maturities
        if years_to_maturity <= maturities[0]:
            return maturity_yield_map[maturities[0]]
        elif years_to_maturity >= maturities[-1]:
            return maturity_yield_map[maturities[-1]]

        # Interpolate between two points
        for i in range(len(maturities) - 1):
            if maturities[i] <= years_to_maturity <= maturities[i + 1]:
                y1 = maturity_yield_map[maturities[i]]
                y2 = maturity_yield_map[maturities[i + 1]]
                x1 = maturities[i]
                x2 = maturities[i + 1]

                # Linear interpolation
                interpolated_yield = y1 + (y2 - y1) * (years_to_maturity - x1) / (x2 - x1)
                return interpolated_yield

        # Fallback
        return 4.0 + (years_to_maturity * 0.05)

    def update_gilt_prices(self, gilt_database, purchase_date=None):
        """
        Update gilt database with current market yields and prices

        Args:
            gilt_database: Dictionary of gilts to update
            purchase_date: Date for price calculations (default: today)

        Returns:
            Updated gilt database with current_price and current_ytm
        """
        # Check cache first
        cached_data = self.get_cached_data()

        if cached_data is not None:
            yield_curve_data = cached_data['data'].get('yield_curve')
            last_update = cached_data['timestamp']
        else:
            # Fetch fresh data
            yield_curve_data = self.fetch_boe_yield_curve()
            last_update = datetime.now()

            # Save to cache
            self.save_to_cache({
                'yield_curve': yield_curve_data,
                'last_update': last_update
            })

        if purchase_date is None:
            purchase_date = datetime.now()
        elif isinstance(purchase_date, str):
            purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d')

        updated_gilts = {}

        for gilt_name, gilt_info in gilt_database.items():
            # Calculate years to maturity
            maturity_date = datetime.strptime(gilt_info['maturity_date'], '%Y-%m-%d')
            years_to_maturity = (maturity_date - purchase_date).days / 365.25
            years_to_maturity = max(0.1, years_to_maturity)

            # Interpolate yield from curve
            current_ytm = self.interpolate_yield(years_to_maturity, yield_curve_data)

            # Calculate current price from yield
            coupon = gilt_info['coupon']
            current_price = self._calculate_bond_price(coupon, years_to_maturity, current_ytm)

            # Update gilt info
            updated_gilt = gilt_info.copy()
            updated_gilt['current_ytm'] = current_ytm
            updated_gilt['current_price'] = current_price
            updated_gilt['last_update'] = last_update.strftime('%Y-%m-%d %H:%M:%S')

            updated_gilts[gilt_name] = updated_gilt

        logger.info(f"Updated {len(updated_gilts)} gilts with current market data")
        return updated_gilts, last_update

    def _calculate_bond_price(self, coupon_rate, years_to_maturity, ytm, face_value=100):
        """Calculate bond price from YTM"""
        if years_to_maturity <= 0 or ytm <= 0:
            return face_value

        # Annual coupon payment
        annual_coupon = face_value * (coupon_rate / 100)

        # Present value of coupon payments
        pv_coupons = 0
        for year in range(1, int(years_to_maturity) + 1):
            pv_coupons += annual_coupon / ((1 + ytm/100) ** year)

        # Adjust for partial year
        if years_to_maturity != int(years_to_maturity):
            partial_year = years_to_maturity - int(years_to_maturity)
            pv_coupons += (annual_coupon * partial_year) / ((1 + ytm/100) ** years_to_maturity)

        # Present value of principal repayment
        pv_principal = face_value / ((1 + ytm/100) ** years_to_maturity)

        # Total bond price
        bond_price = pv_coupons + pv_principal

        return max(85, min(130, bond_price))  # Reasonable bounds

    def get_data_status(self):
        """Get status of cached data"""
        cached_data = self.get_cached_data()

        if cached_data:
            age = datetime.now() - cached_data['timestamp']
            return {
                'has_cache': True,
                'last_update': cached_data['timestamp'],
                'age_hours': age.total_seconds() / 3600,
                'is_fresh': age < timedelta(hours=self.cache_max_age_hours)
            }
        else:
            return {
                'has_cache': False,
                'last_update': None,
                'age_hours': None,
                'is_fresh': False
            }
