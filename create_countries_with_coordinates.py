# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:58:56 2026

@author: win11
"""

import pandas as pd
import pycountry
from geopy.geocoders import Nominatim
from time import sleep

def create_country_csv_with_geopy():
    """Create country CSV using pycountry for codes and geopy for coordinates"""
    
    geolocator = Nominatim(user_agent="country_coords")
    
    countries_data = []
    
    for country in pycountry.countries:
        try:
            # Get coordinates
            location = geolocator.geocode(country.name, timeout=10)
            sleep(1)  # Respect rate limits
            
            lat = location.latitude if location else None
            lon = location.longitude if location else None
            
            countries_data.append({
                'name': country.name,
                'iso_alpha_2': country.alpha_2,
                'iso_alpha_3': country.alpha_3,
                'latitude': lat,
                'longitude': lon
            })
            
            print(f"✓ {country.name}")
            
        except Exception as e:
            print(f"✗ {country.name}: {e}")
            countries_data.append({
                'name': country.name,
                'iso_alpha_2': country.alpha_2,
                'iso_alpha_3': country.alpha_3,
                'latitude': None,
                'longitude': None
            })
    
    df = pd.DataFrame(countries_data)
    df.to_csv('countries_with_coords.csv', index=False)
    print(f"\n✓ Created CSV with {len(df)} countries")
    return df



dfc = create_country_csv_with_geopy()