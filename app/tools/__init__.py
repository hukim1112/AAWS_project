from app.tools.rag import (
    search_bok_reports_basic, 
    search_bok_reports_self_query, 
    search_bok_reports_multimodal
)
from app.tools.utility import (
    read_image_and_analyze, 
    web_search_custom_tool
)

# Export Tool Lists for Agents
tools_basic = [search_bok_reports_basic]
tools_self_query = [search_bok_reports_self_query]
tools_multimodal = [search_bok_reports_multimodal, read_image_and_analyze, web_search_custom_tool]
