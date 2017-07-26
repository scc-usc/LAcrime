#!/bin/bash
gawk -- '
BEGIN{
    FPAT="([^,\"]*)|(\"((\"\")*[^\"]*)*\")"
}
{
    if (substr($c, 1, 1) == "\"") {
        $c = substr($c, 2, length($c) - 2) # Get text within the two quotes
        gsub("\"\"", "\"", $c)  # Normalize double quotes
    }
    print $c
}
' c=5 < tweets_24may_night.csv