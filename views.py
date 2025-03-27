from flask import Blueprint
#from projectdraft import load_houses
home = Blueprint(__name__, 'home')


@views.route("/")
def home():
    return "This is the home page"


@views.route("/chatbot")
def chatbot():
    # for house1 in house_list:
    #     for house2 in house_list:
    #         weight = house1.generate_edge_weight(house2)
    #         if house1.id != house2.id and weight > 0.5:
    #             houses_bba_graph.add_edge(weight, house1, house2)

    return "houses_graph"
