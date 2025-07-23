These datasets are fundamental because they provide the essential real-world context and complexity needed for a project aiming to apply quantum optimization techniques to the Vehicle Routing Problem (VRP). By focusing on waste collection in the City of Casey, these files, categorized into main spatial data and constraints, offer the crucial real-world information that allows for the creation of a practical and robust VRP model, enabling meaningful evaluation of quantum solutions against classical approaches.

### Main Spatial Data

These datasets provide the core geographical information necessary to define the VRP instance:

* `waste-facility-locations.csv`
    * **Source**: City of Casey Open Data Portal
    * **Explanation**: This file contains the coordinates of waste collection facilities. In the VRP model, these facilities serve as the depots where vehicles start and end their routes.

* `waste-collection-area.csv`
    * **Source**: City of Casey Open Data Portal
    * **Explanation**: This dataset outlines the geographical areas designated for waste collection. It's crucial for understanding the comprehensive coverage required for waste pickup points across the city and is combined with public litter bin data to define customer nodes.

* `public-litter-bins.csv`
    * **Source**: City of Casey Open Data Portal
    * **Explanation**: This file provides the locations of public litter bins. These individual bin locations represent specific customer nodes where waste collection must be performed.

* `road-responsibility.csv`
    * **Source**: City of Casey Open Data Portal
    * **Explanation**: This dataset likely details the road network within the City of Casey, including information about road ownership or maintenance. This data is essential for constructing the routes or paths vehicles will travel between depots and collection points, forming the edges of the VRP graph.

### Constraints

To enhance the realism and accuracy of the VRP model, additional real-world constraints are introduced, which are used to modify the edge weights (distances/travel times) in the VRP graph through classical preprocessing:

* Traffic Volume (`traffic-volume-survey-copy.csv`)
    * **Source**: City of Casey Open Data Portal
    * **Explanation**: This dataset contains information on traffic volumes at various locations. Higher traffic volumes will be used to increase the travel time (and thus the "cost" or weight) of specific road segments, reflecting real-world congestion and its impact on route efficiency.

* Elevation (`caseylga_boundary.zip`, `Order_29EYKH.zip`)
    * **Source**: City of Casey Open Data Portal & Data Victoria
    * **Explanation**: These files provide elevation data and the boundary of the Casey LGA. While not directly a "traffic" constraint, elevation can influence vehicle fuel consumption and speed, potentially affecting travel times and route costs. This data helps in creating a more accurate representation of the terrain.

* Rainfall (`rainfall-data.csv`)
    * **Source**: City of Casey Open Data Portal
    * **Explanation**: Rainfall data can impact road conditions, leading to slower travel speeds and increased risks. This dataset will be used to adjust route costs based on historical or predicted rainfall, making the model more robust to varying weather conditions.