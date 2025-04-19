# Mapping of emergency keywords to YouTube video links,
# including separate entries for children when appropriate.
FIRST_AID_VIDEO_LINKS = {
    "choking": "https://youtu.be/3G_0YxMEViE?si=2pi3ElO_VuZ2WkLI",
    "child choking": "https://youtu.be/4j329wUsl3s?si=pkYbBHDwKtkDhZ8n",
    "seizure": "https://youtu.be/4qWPFCFmRlI?si=GJ6RphbEI1JPb7cL",
    "bleeding": "https://youtu.be/NxO5LvgqZe0?si=OFaPVCtMUjy_NGY6",
    "nosebleed": "https://youtu.be/PmmhxW0vVXA?si=hreePm9xgmeSehzz",
    "stroke": "https://youtu.be/EYUDS3wVWEk?si=dDmzdsNqVJn-0S1f",
    "overdose": "https://youtu.be/iaP-Vf4y3fA?si=-0Rd46pn2WHIoh1R",

    # New topics (fill in links)
    "burns": "https://youtu.be/XJGPzl3ENKo?si=o0wgVzSuMX9C4F3Z",
    "child burns": "https://youtu.be/nbJhyap4zP0?si=wHoG2ney__ttPnLx",
    "cpr": "https://youtu.be/Plse2FOkV4Q?si=8DcI6sfrfqOk-7ey",
    "child cpr": "https://youtu.be/c7Q1s7ppSwc?si=rLm9QgfzzPbFX7v-",
    "infant cpr": "https://youtu.be/n65HW1iJUuY?si=0PkUqokCUAzvGW4C",
    "aed": "https://youtu.be/2VxVQ2expR0?si=plyjx8oALqR-evTJ",
    "child aed": "https://youtu.be/2dP1WOa1jGY?si=AEnV-47zDcsSioRs",
    "heat-stroke": "https://youtu.be/jvGC_dQJUtE?si=EEYPIQN89GpXN3Cj",
    "hypothermia": "https://youtu.be/_doDuU-FzTY?si=LdcJMawyxbfUqPoZ",
    "anaphylaxis": "https://youtu.be/QL_CD0-uEL0?si=0GO-Bv13FTebhMEw",
    "child anaphylaxis": "https://youtu.be/U__mH8zaOGY?si=9Vf9OUOOVfNWOlpQ",
    "concussion": "https://youtu.be/vbpisnxsdts?si=p9ZB3gKay_Ml5xnu",
    "child concussion": "https://youtu.be/XP92cyXy03o?si=5DwqTbUL1_IHqX2d",
    "fracture": "https://youtu.be/2v8vlXgGXwE?si=51q5ZPX71nvUp6kg",
    "splinting": "https://youtu.be/jhXWT4UpC-8?si=DzaPaORopVoy_7HG",
    "snake bite": "https://youtu.be/lLkw4BXa7pQ?si=64CBf58NYBZssVPp",
    "bee sting": "https://youtu.be/FqCIp-eJHCg?si=-waKZQu0Hm5Hn5ns",
    "child bee sting": "https://youtu.be/UAuHVKnXhjI?si=ftSN8v-jKS_6shdK",
    "drowning": "https://youtu.be/Hlrbio-NpxQ?si=8J_LnwkPRj7mER06"
}


def get_first_aid_video(query: str) -> str:
    """
    Determines if the query includes an emergency keyword and returns the corresponding first aid video link.
    Prioritize child-related emergencies if the query mentions "child" or "children".
    """
    query_lower = query.lower()
    
    # Check for children-specific emergencies
    if "child" in query_lower or "children" in query_lower:
        if "choking" in query_lower:
            # Return the children-specific choking video link
            return FIRST_AID_VIDEO_LINKS.get("child choking")
    
    # Check the rest of the keywords
    for keyword, video_url in FIRST_AID_VIDEO_LINKS.items():
        # If keyword is found in the query, return the link
        if keyword in query_lower:
            return video_url
    return None
