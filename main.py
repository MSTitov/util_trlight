import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

def fill_sector(ax, distance, theta):
    """
    Попытка добавить в график ax заливку секторов... Уже не помню зачем делал.

    :param ax: исходный график
    :param distance: лист радиусов
    :param theta: лист азимутов
    :return:
    """
    for i in range(len(theta)):
        ax.fill_between(
            np.linspace(theta[i - 1], theta[i], 50),  # Go from 0 to pi/2
            0,  # Fill from radius 0
            150,  # To radius 1
            alpha=0.2,
            color='g',
        )
    return


def draw_lines(ax, distance, theta):
    """
    Добавляет в график линии от начала координат до точек (r,theta)

    :param ax: исходный график
    :param distance: лист радиусов
    :param theta: лист азимутов
    :return:
    """

    for i in range(len(theta)):
        ax.vlines(theta[i], 0, distance[i], colors='g', zorder=2)
    return

def draw_car(ax):
    """
    Добавляет в график ax прямоугольник по размерам Lada Vesta

    :param ax: исходный график
    :return:
    """
    car_points = [(0.85, 2.2), (-0.85, 2.2), (-0.85, -2.2), (0.85, -2.2), (0.85, 2.2)]
    theta = []
    distance = []

    for point in car_points:
        rho, phi = cart_to_pol(point)
        distance.append(rho)
        theta.append(phi)

    ax.plot(theta, distance, 'r')


def cart_to_pol(point):
    """
    Конвертация точки из прямоугольной СК в полярную

    :param point: лист с точкой (x,y)
    :return:
    """
    rho = np.sqrt(point[0] ** 2 + point[1] ** 2)
    phi = np.arctan2(point[1], point[0])
    return (rho, phi)


def draw_intersection(ax):
    """
    Добавляет в исходный график ax визуализацию перекрестка. X образный перекресток по две полосы в каждом направлении

    :param ax: исходный график
    :return:
    """
    quarter_intersection = np.array(
        [(100, 0), (8, 0), (8, -4), (100, -4), (100, -8), (8, -8), (8, -100), (4, -100), (4, -8), (0, -8), (0, -100)])

    car_pose = np.array([-6, 18])
    alfa = np.deg2rad(90)

    rot = np.array([[np.cos(alfa), -np.sin(alfa)], [np.sin(alfa), np.cos(alfa)]])

    for k in range(4):
        distance = []
        theta = []
        alfa = np.deg2rad(90) * k
        rot = np.array([[np.cos(alfa), -np.sin(alfa)], [np.sin(alfa), np.cos(alfa)]])

        points = []
        for i in range(len(quarter_intersection)):
            points.append(np.dot(rot, quarter_intersection[i]) + car_pose)

        for point in points:
            rho, phi = cart_to_pol(point)
            distance.append(rho)
            theta.append(phi)

        ax.plot(theta, distance, 'b')

    return


def visualization():
    # данные об азимутах и дистанциях из таблицы
    distance = np.array([140, 140, 140, 102, 102, 102, 140, 140, 140, 120, 120, 120, 140])
    theta = np.array([15, 25, 35, 70, 90, 110, 145, 155, 165, 250, 270, 290, 15])

    theta_default = range(0, 360, 45)
    theta_all = [*theta_default, *theta]

    theta = np.dot(theta, np.pi * 1 / 180)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # set the locations and labels of the angular gridlines
    lines, labels = plt.thetagrids(theta_all)

    # Отрисовывает distance и theta в виде зеленных точек
    ax.plot(theta, distance, 'g.', markersize=10)
    # Отрисовывает контур по distance и theta
    ax.plot(theta, distance)
    # fill_sector(ax, distance, theta)
    draw_lines(ax, distance, theta)
    draw_car(ax)

    # draw_intersection(ax)

    ax.set_rmax(150)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    pos = ax.get_rlabel_position()
    ax.set_rlabel_position(pos + 7)
    ax.grid(True)

    ax.set_title("Maximum visibility range", va='bottom')
    # plt.show()

def unitVector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def centerline_viz(data_frame):
    azimut = np.array([])
    elevation = np.array([])
    radius = np.array([])

    for index, row_lane in data_frame.iterrows():
        x_lane = pd.array(row_lane["x_lane"])
        y_lane = pd.array(row_lane["y_lane"])
        z_lane = pd.array(row_lane["z_lane"])

        u_lane = x_lane[1:] - x_lane[:-1]
        v_lane = y_lane[1:] - y_lane[:-1]
        k_lane = z_lane[1:] - z_lane[:-1]

        u_lane = np.append(u_lane, u_lane[-1])
        v_lane = np.append(v_lane, v_lane[-1])
        k_lane = np.append(k_lane, k_lane[-1])

        x_normal = pd.array(row_lane["x_normal"])
        y_normal = pd.array(row_lane["y_normal"])
        z_normal = pd.array(row_lane["z_normal"])

        u_normal = x_normal - x_lane
        v_normal = y_normal - y_lane
        k_normal = z_normal - z_lane

        matrix_invers = np.ones(shape=[len(u_lane), 3, 3])
        orto_angle = []

        for i in range(len(u_lane)):
            vector_lane = unitVector(np.array([u_lane[i], v_lane[i], k_lane[i]]))
            vector_normal = unitVector(np.array([u_normal[i], v_normal[i], k_normal[i]]))

            y_axis = np.cross(vector_normal, vector_lane)
            orto_angle.append(np.arccos(np.dot(vector_normal, vector_lane)) / np.pi * 180)
            vector_normal = np.cross(vector_lane, y_axis)

            matrix_invers[i] = np.matrix([[vector_lane[0], y_axis[0], vector_normal[0]],
                                          [vector_lane[1], y_axis[1], vector_normal[1]],
                                          [vector_lane[2], y_axis[2], vector_normal[2]]]).getI()

        min_angle_between_normal_and_lane = min(orto_angle)
        max_angle_between_normal_and_lane = max(orto_angle)
        print("Min: ", min_angle_between_normal_and_lane , " Max: ", max_angle_between_normal_and_lane)

        if min_angle_between_normal_and_lane < 89 or max_angle_between_normal_and_lane > 93:
            print("Error for ", row_lane["lane_id"])
            continue


        # Add points to figure
        for df_tl in row_lane["lane_to_tl"]:
            x_tl = df_tl["x_tl"]
            y_tl = df_tl["y_tl"]
            z_tl = df_tl["z_tl"]

            u_light = np.array([x_tl] * len(x_lane)) - x_lane
            v_light = np.array([y_tl] * len(y_lane)) - y_lane
            k_light = np.array([z_tl] * len(z_lane)) - z_lane

            r = np.sqrt(u_light ** 2 + v_light ** 2 + k_light ** 2)
            radius = np.append(radius, r)

            tl_new = np.ones(shape=[len(u_lane), 3, 1])
            for i in range(len(u_lane)):
                tl_new[i] = matrix_invers[i] * np.matrix([[u_light[i]], [v_light[i]], [k_light[i]]])

            for i in range(len(u_lane)):
                vector = unitVector(tl_new[i])

                elevation_new = 90 - np.arccos(vector[2])/ np.pi * 180
                azimut_new = -np.arctan2(vector[1], vector[0])/ np.pi * 180
                elevation = np.append(elevation, elevation_new)
                azimut = np.append(azimut, azimut_new)

    fig = px.scatter(x = azimut, y = elevation, labels={'x':'azimut', 'y':'elevation', 'color':'distance'}, color=list(radius), template="plotly_dark", color_continuous_scale=px.colors.sequential.Agsunset)
    fig.show()

    ########################################################################
    # Make background layout
    ########################################################################
    # https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Barpolar.html
    theta = [95, 95, 90]
    width = [91, 28, 103]
    vals = [40., 100., 40.]
#    blue = rgba(50, 144, 196, 0.2) = Governance
#    orange = rgba(183, 87, 39, 0.2) = Design
#    yellow = rgba(217, 166, 25, 0.2) = Implementation
#    green = rgba(55, 121, 62, 0.2) = Verification
#    brown = rgba(121, 31, 23, 0.2) = Operations
    colors = ['rgba(55, 121, 62, 0.2)', 'rgba(217, 166, 25, 0.2)', 'rgba(121, 31, 23, 0.2)']
    labels = ["CF",  "Zed", "CF_short"]
    barpolar_plots = [go.Barpolar(r=[r], theta=[t], width=[w], name=n, marker_color=[c])
                      for r, t, w, n, c in zip(vals, theta, width, labels, colors)]
    layout = go.Figure()

    # Align backgorund
    angular_tickvals = []
    layout.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 130], showticklabels=False, visible = False, ticks=''),
            angularaxis = dict(showticklabels=False, ticks='')
        ), polar_angularaxis_tickvals=angular_tickvals
    )
    # Remove legends
    layout_options = {}
    layout_options["showlegend"] = True
    layout_options["polar_angularaxis_showticklabels"] = False
    layout.update_layout(**layout_options)
    ########################################################################
    ########################################################################
    # Add plots on Layout
    ########################################################################
    layout.add_traces(barpolar_plots)

    fig = px.scatter_polar(r=radius, theta=azimut)
    fig.update_traces(subplot="polar2")

    layout = layout.add_traces(fig.data).update_layout({ax:{"domain":{"x":[0,1]}} for ax in ["polar","polar2"]})
    layout.update_layout(polar2={"bgcolor":"rgba(0,0,0,0)"})
    layout.update_layout(
        template=None,
        polar2 = dict(
          radialaxis_tickfont_size = 12,
          angularaxis = dict(
            tickfont_size = 12,
            rotation = 90,
            direction = "clockwise"
        )
    ))

    layout.show()

    width = [65, 23, 71]
    barpolar_plots = [go.Barpolar(r=[r], theta=[t], width=[w], name=n, marker_color=[c])
                      for r, t, w, n, c in zip(vals, theta, width, labels, colors)]
    layout = go.Figure()

    # Align backgorund
    angular_tickvals = []
    layout.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0, 130], showticklabels=False, visible = False, ticks=''),
            angularaxis = dict(showticklabels=False, ticks='')
        ), polar_angularaxis_tickvals=angular_tickvals
    )
    # Remove legends
    layout_options = {}
    layout_options["showlegend"] = True
    layout_options["polar_angularaxis_showticklabels"] = False
    layout.update_layout(**layout_options)
    ########################################################################
    ########################################################################
    # Add plots on Layout
    ########################################################################
    layout.add_traces(barpolar_plots)

    fig = px.scatter_polar(r=radius, theta=elevation)
    fig.update_traces(subplot="polar2")

    layout = layout.add_traces(fig.data).update_layout({ax:{"domain":{"x":[0,1]}} for ax in ["polar","polar2"]})
    layout.update_layout(polar2={"bgcolor":"rgba(0,0,0,0)"})
    layout.update_layout(
        template=None,
        polar = dict(
          radialaxis_tickfont_size = 12,
          angularaxis = dict(
            tickfont_size = 12,
            rotation = 270
        )
    ))

    layout.show()

    return

def main():
    # visualization()

    # Загрузка данных
    data_frame = pd.read_json("vector_map_element_poses.json")
    centerline_viz(data_frame)
    
    return

if __name__ == '__main__':
    main()
