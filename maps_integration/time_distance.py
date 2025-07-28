import openrouteservice
import folium
import webbrowser
from datetime import datetime, timedelta

def main():
    # Load your ORS API key
    with open("maps_integration/key.txt") as f:
        ors_key = f.read().strip()

    client = openrouteservice.Client(key=ors_key)

    # User input address (string)
    user_address = input("Enter your current address or location: ").strip()

    # Use ORS geocode (Pelias) to get origin coords
    try:
        geocode_result = client.pelias_search(user_address)
        features = geocode_result.get('features')
        if not features:
            print("Address not found, please enter a more specific address.")
            return
        origin_coords = tuple(features[0]['geometry']['coordinates'])  # (lon, lat)
    except Exception as e:
        print("Geocoding error:", e)
        return

    # Validate coords roughly for India (optional)
    lon, lat = origin_coords
    if not (68 <= lon <= 98 and 6 <= lat <= 38):
        print(f"Geocoded coordinates ({lon}, {lat}) are outside India. Please be more specific.")
        return

    # Fixed destination coords (lon, lat)
    destination_coords = (73.86567022368467, 18.531292153194457)

    print(f"Origin coords: {origin_coords}")
    print(f"Destination coords: {destination_coords}")

    # Get directions from ORS
    try:
        route = client.directions(
            coordinates=[origin_coords, destination_coords],
            profile='driving-car',
            format='geojson'  # Use geojson for easy plotting
        )
    except openrouteservice.exceptions.ApiError as e:
        print("Routing API error:", e)
        return

    summary = route['features'][0]['properties']['summary']
    distance_km = summary['distance'] / 1000
    duration_min = summary['duration'] / 60

    now = datetime.now()
    arrival_time = now + timedelta(seconds=summary['duration'])

    print("\nRoute Information:")
    print(f"Distance: {distance_km:.2f} km")
    print(f"Estimated travel time: {duration_min*2:.1f} minutes")
    print(f"Current time: {now.strftime('%H:%M:%S')}")
    print(f"Estimated arrival time: {arrival_time.strftime('%H:%M:%S')}")

    # Create a folium map centered at midpoint between origin and destination
    mid_lat = (origin_coords[1] + destination_coords[1]) / 2
    mid_lon = (origin_coords[0] + destination_coords[0]) / 2
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=12)

    # Add origin marker
    folium.Marker(
        location=[origin_coords[1], origin_coords[0]],
        popup="Origin",
        icon=folium.Icon(color='green')
    ).add_to(m)

    # Add destination marker
    folium.Marker(
        location=[destination_coords[1], destination_coords[0]],
        popup="Destination",
        icon=folium.Icon(color='red')
    ).add_to(m)

    # Add route polyline
    coords = route['features'][0]['geometry']['coordinates']  # list of [lon, lat]
    # Convert to [lat, lon] for folium
    latlngs = [(coord[1], coord[0]) for coord in coords]
    folium.PolyLine(latlngs, color='blue', weight=5, opacity=0.7).add_to(m)

    # Save map to an HTML file
    map_file = "route_map.html"
    m.save(map_file)

    print(f"\nMap saved to {map_file}, opening in web browser...")
    webbrowser.open(map_file)  # Open the HTML file in the default browser

        # Prepare Google Maps navigation link
    origin_lat, origin_lon = origin_coords[1], origin_coords[0]
    dest_lat, dest_lon = destination_coords[1], destination_coords[0]
    gmaps_url = (
        f"https://www.google.com/maps/dir/?api=1"
        f"&origin={origin_lat},{origin_lon}"
        f"&destination={dest_lat},{dest_lon}"
        f"&travelmode=driving"
    )
    print(f"\nGoogle Maps navigation link:\n{gmaps_url}")

    return arrival_time.strftime('%H%M%S')



if __name__ == "__main__":
    main()
