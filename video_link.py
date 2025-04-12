# Mapping of emergency keywords to YouTube video links,
# including separate entries for children when appropriate.
FIRST_AID_VIDEO_LINKS = {
    "choking": "https://youtu.be/3G_0YxMEViE?si=2pi3ElO_VuZ2WkLI",
    "child choking": "https://youtu.be/4j329wUsl3s?si=pkYbBHDwKtkDhZ8n",
    "seizure": "https://youtu.be/4qWPFCFmRlI?si=GJ6RphbEI1JPb7cL",
    "bleeding": "https://youtu.be/NxO5LvgqZe0?si=OFaPVCtMUjy_NGY6",  # General bleeding
    "nosebleed": "https://youtu.be/V34aB5uheBg?si=q6V8Q8EGb_-5t2vh",   # Nosebleeds
    "stroke": "https://youtu.be/EYUDS3wVWEk?si=dDmzdsNqVJn-0S1f",     # Stroke recognition and response
    "overdose": "https://youtu.be/iaP-Vf4y3fA?si=-0Rd46pn2WHIoh1R"    # Opioid overdose response (example)
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
