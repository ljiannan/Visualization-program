#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aerial Photography Keywords Collection
Including categories: Natural Landscapes, Urban Scenes, Special Scenes, etc.
"""

# Natural Landscape Aerial Keywords
NATURE_AERIAL_KEYWORDS = [
    "aerial mountain range", "aerial forest view", "aerial coastline", "aerial river view", "aerial waterfall",
    "aerial grassland", "aerial desert landscape", "aerial lake view", "aerial glacier", "aerial volcano",
    "aerial canyon", "aerial beach", "aerial island view", "aerial snow mountain", "aerial rainforest",
    "aerial coral reef", "aerial wetland", "aerial oasis", "aerial plateau", "aerial hills",
    "aerial stream", "aerial bay view", "aerial wilderness", "aerial cliff", "aerial valley",
    "drone mountain footage", "drone forest footage", "drone coast footage", "drone river footage", "drone waterfall footage",
    "drone nature footage", "aerial nature photography", "aerial landscape photography", "drone landscape footage",
    "aerial mountain photography", "aerial forest photography", "aerial water photography"
]

# Urban Landscape Aerial Keywords
CITY_AERIAL_KEYWORDS = [
    "aerial city skyline", "aerial modern buildings", "aerial historic buildings", "aerial city park",
    "aerial commercial district", "aerial residential area", "aerial industrial zone", "aerial harbor", "aerial airport",
    "aerial train station", "aerial stadium view", "aerial university campus", "aerial city square",
    "aerial urban traffic", "aerial bridge view", "aerial landmark building", "aerial night scene", "aerial urban green",
    "aerial waterfront", "aerial urban planning", "aerial business district", "aerial cultural center",
    "aerial urban infrastructure", "aerial urban renewal", "aerial city block",
    "drone city footage", "drone urban footage", "drone cityscape", "aerial downtown", "aerial metropolis",
    "aerial urban view", "drone city view", "aerial city photography", "drone urban photography"
]

# Agricultural Landscape Aerial Keywords
AGRICULTURE_AERIAL_KEYWORDS = [
    "aerial farmland", "aerial terraced fields", "aerial orchard view", "aerial tea plantation", "aerial rice fields",
    "aerial flower fields", "aerial vineyard", "aerial farm view", "aerial ranch", "aerial greenhouse",
    "aerial irrigation system", "aerial agricultural base", "aerial breeding farm", "aerial agri-tech park",
    "aerial tourism farm", "aerial eco-agriculture", "aerial organic farm", "aerial modern agriculture",
    "aerial specialty farming", "aerial agricultural park",
    "drone farm footage", "drone agriculture footage", "aerial farming", "drone crop fields"
]

# Special Scene Aerial Keywords
SPECIAL_AERIAL_KEYWORDS = [
    "aerial sunrise view", "aerial sunset view", "aerial fog view", "aerial clouds view", "aerial aurora",
    "aerial starry sky", "aerial snow scene", "aerial rain scene", "aerial storm view", "aerial rainbow",
    "aerial lightning", "aerial sandstorm", "aerial typhoon", "aerial flood view", "aerial fire scene",
    "aerial earthquake zone", "aerial volcanic eruption", "aerial tsunami", "aerial ice melting", "aerial pollution",
    "drone weather footage", "drone natural phenomena", "aerial weather photography", "drone atmospheric footage"
]

# Cultural Heritage Aerial Keywords
HERITAGE_AERIAL_KEYWORDS = [
    "aerial ancient city", "aerial historic town", "aerial temple view", "aerial palace view", "aerial great wall",
    "aerial ancient tomb", "aerial archaeological site", "aerial cultural protection", "aerial historic district",
    "aerial ancient architecture", "aerial cultural heritage", "aerial ancient village", "aerial pagoda",
    "aerial city wall", "aerial ancient garden", "aerial ancient bridge", "aerial historic post",
    "aerial ancient battlefield", "aerial ancient port", "aerial ancient path",
    "drone heritage footage", "drone historical site", "aerial monument view", "drone cultural footage"
]

# Transportation Facility Aerial Keywords
TRANSPORTATION_AERIAL_KEYWORDS = [
    "aerial highway view", "aerial railway track", "aerial road interchange", "aerial tunnel entrance", "aerial dock view",
    "aerial airport runway", "aerial metro station", "aerial transit hub", "aerial parking lot",
    "aerial logistics center", "aerial high-speed rail", "aerial road network", "aerial shipping port",
    "aerial transportation hub", "aerial rail transit", "aerial passenger station", "aerial freight station",
    "aerial traffic control", "aerial toll station", "aerial service area",
    "drone transport footage", "drone traffic footage", "aerial infrastructure", "drone highway footage"
]

# Industrial Facility Aerial Keywords
INDUSTRIAL_AERIAL_KEYWORDS = [
    "aerial factory view", "aerial power plant", "aerial mining site", "aerial steel plant", "aerial chemical plant",
    "aerial industrial park", "aerial storage facility", "aerial manufacturing base", "aerial energy facility",
    "aerial industrial port", "aerial industrial wasteland", "aerial industrial ruins", "aerial industrial zone",
    "aerial high-tech park", "aerial eco-industrial", "aerial circular economy",
    "aerial industrial cluster", "aerial industrial city", "aerial science park", "aerial innovation park",
    "drone industrial footage", "drone factory footage", "aerial industrial photography", "drone manufacturing footage"
]

# Sports Facility Aerial Keywords
SPORTS_AERIAL_KEYWORDS = [
    "aerial stadium view", "aerial football field", "aerial basketball court", "aerial tennis court", "aerial swimming pool",
    "aerial sports center", "aerial athletic field", "aerial fitness park", "aerial ski resort",
    "aerial golf course", "aerial equestrian field", "aerial racing track", "aerial bicycle path",
    "aerial extreme sports", "aerial water sports", "aerial sports training",
    "aerial sports park", "aerial sports town", "aerial fitness center", "aerial sports complex",
    "drone sports footage", "drone stadium footage", "aerial sports photography", "drone athletics footage"
]

# Tourism Spot Aerial Keywords
TOURISM_AERIAL_KEYWORDS = [
    "aerial scenic area", "aerial theme park", "aerial resort view", "aerial amusement park", "aerial tourist area",
    "aerial tourist attraction", "aerial scenic spot", "aerial holiday resort", "aerial eco-tourism",
    "aerial cultural tourism", "aerial leisure resort", "aerial tourist town", "aerial tourist street",
    "aerial scenic view", "aerial tourism landscape", "aerial tourist center", "aerial destination",
    "aerial tourism park", "aerial tourism district", "aerial tourism complex",
    "drone tourism footage", "drone travel footage", "aerial tourist photography", "drone vacation footage"
]

# All aerial keywords collection
ALL_AERIAL_KEYWORDS = (
    NATURE_AERIAL_KEYWORDS +
    CITY_AERIAL_KEYWORDS +
    AGRICULTURE_AERIAL_KEYWORDS +
    SPECIAL_AERIAL_KEYWORDS +
    HERITAGE_AERIAL_KEYWORDS +
    TRANSPORTATION_AERIAL_KEYWORDS +
    INDUSTRIAL_AERIAL_KEYWORDS +
    SPORTS_AERIAL_KEYWORDS +
    TOURISM_AERIAL_KEYWORDS
)

# Get total number of keywords
TOTAL_KEYWORDS = len(ALL_AERIAL_KEYWORDS)

if __name__ == "__main__":
    print(f"Total collected {TOTAL_KEYWORDS} aerial keywords")
    print("\nKeyword count by category:")
    print(f"Natural Landscape: {len(NATURE_AERIAL_KEYWORDS)}")
    print(f"Urban Landscape: {len(CITY_AERIAL_KEYWORDS)}")
    print(f"Agricultural Landscape: {len(AGRICULTURE_AERIAL_KEYWORDS)}")
    print(f"Special Scene: {len(SPECIAL_AERIAL_KEYWORDS)}")
    print(f"Cultural Heritage: {len(HERITAGE_AERIAL_KEYWORDS)}")
    print(f"Transportation Facility: {len(TRANSPORTATION_AERIAL_KEYWORDS)}")
    print(f"Industrial Facility: {len(INDUSTRIAL_AERIAL_KEYWORDS)}")
    print(f"Sports Facility: {len(SPORTS_AERIAL_KEYWORDS)}")
    print(f"Tourism Spot: {len(TOURISM_AERIAL_KEYWORDS)}") 