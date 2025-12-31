"""
Quick test script for bond_data_fetcher module
"""

from bond_data_fetcher import BondDataFetcher
from datetime import datetime

print("Testing BondDataFetcher...")
print("=" * 60)

# Create fetcher instance
fetcher = BondDataFetcher()

print("\n1. Testing Bank of England yield curve fetch...")
try:
    yield_data = fetcher.fetch_boe_yield_curve()
    if yield_data is not None and not yield_data.empty:
        print(f"✅ SUCCESS: Retrieved {len(yield_data)} rows of yield data")
        print(f"   Columns: {list(yield_data.columns)[:5]}...")  # Show first 5 columns
        print(f"   Latest data point: {yield_data.iloc[-1]['DATE'] if 'DATE' in yield_data.columns else 'N/A'}")
    else:
        print("⚠️  WARNING: No yield data returned (may be API issue or network)")
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n2. Testing yield interpolation...")
try:
    # Test interpolating yield for a 5-year bond
    test_yield = fetcher.interpolate_yield(5.0)
    print(f"✅ SUCCESS: 5-year yield = {test_yield:.3f}%")

    # Test 10-year
    test_yield_10 = fetcher.interpolate_yield(10.0)
    print(f"✅ SUCCESS: 10-year yield = {test_yield_10:.3f}%")
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n3. Testing bond price calculation...")
try:
    # Test price for UK Treasury 4.25% 2032 (TR32)
    coupon = 4.25
    ytm = 4.08  # Current market yield from YieldGimp
    years = 7.5  # Approximate

    price = fetcher._calculate_bond_price(coupon, years, ytm)
    print(f"✅ SUCCESS: TR32 calculated price = {price:.2f}")
    print(f"   (YieldGimp shows 100.94, should be close)")
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n4. Testing cache functionality...")
try:
    status = fetcher.get_data_status()
    print(f"✅ Cache status:")
    print(f"   Has cache: {status['has_cache']}")
    if status['has_cache']:
        print(f"   Last update: {status['last_update']}")
        print(f"   Age: {status['age_hours']:.2f} hours")
        print(f"   Is fresh: {status['is_fresh']}")
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n5. Testing gilt database update...")
try:
    # Create a small test gilt database
    test_gilts = {
        'UK Treasury 4.25% 2032': {
            'isin': 'GB0004893086',
            'maturity_date': '2032-06-07',
            'coupon': 4.25,
            'type': 'UK Gilt',
            'rating': 'AA',
        }
    }

    updated_gilts, last_update = fetcher.update_gilt_prices(test_gilts)

    print(f"✅ SUCCESS: Updated gilt database")
    for name, info in updated_gilts.items():
        print(f"   {name}:")
        print(f"     Current YTM: {info.get('current_ytm', 'N/A'):.3f}%")
        print(f"     Current Price: {info.get('current_price', 'N/A'):.2f}")
        print(f"     Last Update: {info.get('last_update', 'N/A')}")
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("\nNote: Some tests may show warnings if the BoE API is unavailable,")
print("but the system will fall back to default estimates.")
