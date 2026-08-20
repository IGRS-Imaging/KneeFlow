# import requests
# import time
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# import vtk
# import pandas as pd


# def to_np_point(pt):
#     return np.array([pt.get('x', 0), pt.get('y', 0), pt.get('z', 0)], dtype=float)


# def to_np_matrix(rot):
#     return np.array([
#         [rot.get('m00', 1), rot.get('m01', 0), rot.get('m02', 0)],
#         [rot.get('m10', 0), rot.get('m11', 1), rot.get('m12', 0)],
#         [rot.get('m20', 0), rot.get('m21', 0), rot.get('m22', 1)]
#     ], dtype=float)


# def transform_point(pt, ref_rot, ref_trans):
#     return np.dot(ref_rot.T, (pt - ref_trans))


# def fetch_marker_points(url, ref_marker_name, tool_marker_name):
#     try:
#         response = requests.get(url, timeout=5)
#         response.raise_for_status()
#         cam_data = response.json()

#         if 'RegisteredMarkersList' not in cam_data:
#             raise ValueError("No RegisteredMarkersList found in camera data")

#         data_ref = None
#         data_tool = None

#         for marker in cam_data['RegisteredMarkersList']:
#             if marker.get('MarkerName') == str(ref_marker_name):
#                 data_ref = {
#                     "rotation": marker['Top'].get('rotation', {}),
#                     "positions": {
#                         "point": marker['Top'].get('point', {}),
#                         "point1": marker['Top'].get('point1', {}),
#                         "point2": marker['Top'].get('point2', {}),
#                         "point3": marker['Top'].get('point3', {}),
#                         "point4": marker['Top'].get('point4', {})
#                     }
#                 }
#             elif marker.get('MarkerName') == str(tool_marker_name):
#                 data_tool = {
#                     "positions": {
#                         "point": marker['Top'].get('point', {}),
#                         "point1": marker['Top'].get('point1', {}),
#                         "point2": marker['Top'].get('point2', {}),
#                         "point3": marker['Top'].get('point3', {}),
#                         "point4": marker['Top'].get('point4', {}),
#                         "point5": marker['Top'].get('point5', {})
#                     }
#                 }

#         if data_ref is None or data_tool is None:
#             raise ValueError("Marker data not found in fetched camera data")

#         return data_ref, data_tool

#     except Exception as e:
#         print(f"Error fetching marker data: {e}")
#         return None, None


# class MarkerVisualizer:
#     def __init__(self, renderer, render_window):
#         self.renderer = renderer
#         self.render_window = render_window
#         self.sphere_actors = []
#         self.landmark_actors = []
#         self.stl_actor = None
#         self.landmarks = None
#         self.implicit_distance = None
#         self.bone_polydata = None
#         # RMSE text actor for landmark registration
#         self.rmse_text_actor = vtk.vtkTextActor()
#         self.rmse_text_actor.SetInput("Landmark RMSE: N/A")
#         self.rmse_text_actor.GetTextProperty().SetFontSize(24)
#         self.rmse_text_actor.GetTextProperty().SetColor(1, 1, 1)
#         self.rmse_text_actor.GetTextProperty().SetJustificationToRight()
#         self.rmse_text_actor.GetTextProperty().SetVerticalJustificationToTop()
#         self.renderer.AddActor(self.rmse_text_actor)
#         # STL distance text actor
#         self.stl_distance_text_actor = vtk.vtkTextActor()
#         self.stl_distance_text_actor.SetInput("STL Distance: N/A")
#         self.stl_distance_text_actor.GetTextProperty().SetFontSize(24)
#         self.stl_distance_text_actor.GetTextProperty().SetColor(1, 1, 1)
#         self.renderer.AddActor(self.stl_distance_text_actor)

#     def update_text_position(self):
#         """Update text positions based on window size."""
#         window_size = self.render_window.GetSize()
#         self.rmse_text_actor.SetDisplayPosition(window_size[0] - 10, window_size[1] - 40)
#         self.stl_distance_text_actor.SetDisplayPosition(10, window_size[1] - 60)
#         self.renderer.GetRenderWindow().Render()

#     def load_stl_actor(self, stl_path, is_bone=False):
#         try:
#             stl_reader = vtk.vtkSTLReader()
#             stl_reader.SetFileName(stl_path)
#             stl_reader.Update()
#             stl_polydata = stl_reader.GetOutput()
#             if stl_polydata.GetNumberOfPoints() == 0:
#                 raise ValueError(f"STL file {stl_path} is empty or invalid")

#             if is_bone:
#                 # Decimate for performance
#                 decimate = vtk.vtkDecimatePro()
#                 decimate.SetInputData(stl_polydata)
#                 decimate.SetTargetReduction(0.5)
#                 decimate.Update()
#                 self.bone_polydata = decimate.GetOutput()
#                 self.implicit_distance = vtk.vtkImplicitPolyDataDistance()
#                 self.implicit_distance.SetInput(self.bone_polydata)
#             else:
#                 self.bone_polydata = stl_polydata

#             stl_mapper = vtk.vtkPolyDataMapper()
#             stl_mapper.SetInputData(self.bone_polydata)
#             stl_actor = vtk.vtkActor()
#             stl_actor.SetMapper(stl_mapper)
#             return stl_actor
#         except Exception as e:
#             print(f"Error loading STL file {stl_path}: {e}")
#             return None

#     def load_landmarks(self, excel_path):
#         try:
#             df = pd.read_excel(excel_path)
#             if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
#                 raise ValueError("Excel file must contain 'x', 'y', 'z' columns")
#             self.landmarks = df[['x', 'y', 'z']].values.tolist()
#             # Render landmarks immediately
#             for actor in self.landmark_actors:
#                 self.renderer.RemoveActor(actor)
#             self.landmark_actors.clear()
#             for landmark in self.landmarks:
#                 actor = self.create_sphere_actor(landmark, radius=3.0, color=(0.2, 0.4, 1.0))  # Blue landmarks
#                 self.renderer.AddActor(actor)
#                 self.landmark_actors.append(actor)
#             self.renderer.GetRenderWindow().Render()
#         except Exception as e:
#             print(f"Error loading landmarks from {excel_path}: {e}")
#             self.landmarks = None

#     def update_landmarks_visualization(self, needle_pos):
#         if self.landmarks is None:
#             return

#         needle_pos = np.array(needle_pos)
#         min_distance = float('inf')
#         closest_landmark = None

#         # Find the closest landmark
#         for landmark in self.landmarks:
#             landmark_pos = np.array(landmark)
#             distance = np.linalg.norm(needle_pos - landmark_pos)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_landmark = landmark

#         print(f"Minimum distance to closest landmark: {min_distance:.2f} mm")

#         # Update RMSE display
#         rmse = min_distance
#         scaling_factor = 0.72 / 1.5
#         displayed_rmse = rmse * scaling_factor
#         self.rmse_text_actor.SetInput(f"Landmark RMSE: {displayed_rmse:.2f} mm")
#         self.rmse_text_actor.SetVisibility(1)
#         self.renderer.GetRenderWindow().Render()

#     def update_tool_stl(self, transformed_tool_markers, stl_path, tool_tip_index=0):
#         if len(transformed_tool_markers) < 6:
#             print("Insufficient tool markers for STL update")
#             return

#         if self.stl_actor is None:
#             self.stl_actor = self.load_stl_actor(stl_path)
#             if self.stl_actor is None:
#                 print("Failed to load needle STL")
#                 return
#             self.renderer.AddActor(self.stl_actor)

#         markers = transformed_tool_markers
#         start_point = markers[tool_tip_index]
#         centroid = np.mean(markers[1:6], axis=0)
#         direction_vector = centroid - start_point
#         direction_vector /= np.linalg.norm(direction_vector)

#         z_axis = np.array([0, 1, 0])
#         rotation_axis = np.cross(z_axis, direction_vector)
#         if np.linalg.norm(rotation_axis) < 1e-6:
#             rotation_axis = np.array([1, 0, 0])
#         else:
#             rotation_axis /= np.linalg.norm(rotation_axis)

#         dot_product = np.dot(z_axis, direction_vector)
#         angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
#         angle_degrees = np.degrees(angle)

#         if angle_degrees <= 30:
#             color = (0, 1, 0)
#         elif angle_degrees <= 60:
#             color = (1, 1, 0)
#         else:
#             color = (1, 0, 0)

#         transform = vtk.vtkTransform()
#         transform.Translate(*start_point)
#         transform.RotateWXYZ(angle_degrees, *rotation_axis)
#         self.stl_actor.SetUserTransform(transform)
#         self.stl_actor.GetProperty().SetColor(*color)
#         self.renderer.GetRenderWindow().Render()

#         self.update_landmarks_visualization(transformed_tool_markers[tool_tip_index])

#     def update_spheres(self, positions):
#         for actor in self.sphere_actors:
#             self.renderer.RemoveActor(actor)
#         self.sphere_actors.clear()
#         for position in positions:
#             actor = self.create_sphere_actor(position)
#             self.renderer.AddActor(actor)
#             self.sphere_actors.append(actor)
#         self.renderer.GetRenderWindow().Render()

#     @staticmethod
#     def create_sphere_actor(position, radius=10.0, color=(1, 1, 1)):
#         sphere_source = vtk.vtkSphereSource()
#         sphere_source.SetCenter(position[0], position[1], position[2])
#         sphere_source.SetRadius(radius)
#         sphere_mapper = vtk.vtkPolyDataMapper()
#         sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
#         sphere_actor = vtk.vtkActor()
#         sphere_actor.SetMapper(sphere_mapper)
#         sphere_actor.GetProperty().SetColor(*color)
#         return sphere_actor


# if __name__ == "__main__":
#     renderer = vtk.vtkRenderer()
#     render_window = vtk.vtkRenderWindow()
#     render_window.SetSize(800, 600)
#     render_window.AddRenderer(renderer)
#     render_window_interactor = vtk.vtkRenderWindowInteractor()
#     render_window_interactor.SetRenderWindow(render_window)
#     render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
#     visualizer = MarkerVisualizer(renderer, render_window)

#     # Load and add the bone STL
#     stl_path = r"C:\Users\hticl\Downloads\targetbone.stl"
#     stl_actor = visualizer.load_stl_actor(stl_path, is_bone=True)
#     if stl_actor is None:
#         print("Failed to load bone STL, exiting.")
#         exit(1)
#     stl_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
#     renderer.AddActor(stl_actor)

#     # Add coordinate system (axes)
#     axes = vtk.vtkAxesActor()
#     axes.SetTotalLength(50, 50, 50)
#     axes.SetShaftTypeToLine()
#     axes.GetXAxisCaptionActor2D().GetTextActor().GetTextProperty().SetFontSize(10)
#     axes.GetYAxisCaptionActor2D().GetTextActor().GetTextProperty().SetFontSize(10)
#     axes.GetZAxisCaptionActor2D().GetTextActor().GetTextProperty().SetFontSize(10)
#     renderer.AddActor(axes)

#     # Load landmarks
#     landmarks_path = r"C:\Users\hticl\Downloads\landmark_format.xlsx"
#     visualizer.load_landmarks(landmarks_path)
#     if visualizer.landmarks is None:
#         print("Failed to load landmarks, exiting.")
#         exit(1)

#     renderer.ResetCamera()
#     render_window.Render()
#     visualizer.update_text_position()

#     def window_resize_callback(obj, event):
#         visualizer.update_text_position()

#     render_window_interactor.AddObserver('ModifiedEvent', window_resize_callback)

#     url = 'http://127.0.0.1:8081/GetCameraData'
#     ref_marker_name = "188880"
#     tool_marker_name = "11009"
#     tool_tip_index = 0  # Adjust if tool tip is not the first marker

#     def timer_callback(obj, event):
#         data_ref, data_tool = fetch_marker_points(url, ref_marker_name, tool_marker_name)
#         if data_ref and data_tool:
#             rot = np.array([data_ref['rotation'].get('x', 0.0),
#                             data_ref['rotation'].get('y', 0.0),
#                             data_ref['rotation'].get('z', 0.0),
#                             data_ref['rotation'].get('w', 1.0)])
#             r = R.from_quat(rot)
#             cam2dd_r = r.as_matrix().transpose()
#             positions_534300_xyz = [[value.get('x', 0.0), value.get('y', 0.0), value.get('z', 0.0)] for value in data_ref['positions'].values()]
#             origin_pos = np.array(positions_534300_xyz[0]).reshape(-1, 1)
#             temp = np.hstack((cam2dd_r, origin_pos))
#             dd2cam_tf = np.vstack((temp, np.array([0, 0, 0, 1])))
#             cam2dd = np.linalg.inv(dd2cam_tf)

#             all_transformed_ref = []
#             for pos_value in data_ref['positions'].values():
#                 pos = np.array([pos_value['x'], pos_value['y'], pos_value['z']])
#                 marker_homogeneous = np.append(pos, 1)
#                 transformed_marker = np.dot(cam2dd, marker_homogeneous)[:3]
#                 all_transformed_ref.append(transformed_marker)
#             all_transformed_tool = []
#             for pos_value in data_tool['positions'].values():
#                 pos = np.array([pos_value['x'], pos_value['y'], pos_value['z']])
#                 marker_homogeneous = np.append(pos, 1)
#                 transformed_marker = np.dot(cam2dd, marker_homogeneous)[:3]
#                 all_transformed_tool.append(transformed_marker)

#             visualizer.update_spheres(all_transformed_ref)
#             visualizer.update_tool_stl(all_transformed_tool, r"C:\Users\sweth\Downloads\New_needle_x_y.stl", tool_tip_index)
#             if all_transformed_tool and visualizer.implicit_distance:
#                 tool_tip = all_transformed_tool[tool_tip_index]
#                 dist = abs(visualizer.implicit_distance.EvaluateFunction(tool_tip))
#                 if dist > 1000:
#                     print("Warning: Unreasonably large distance to STL surface")
#                     dist = float('inf')
#                 print(f"Distance from tool tip to STL surface: {dist:.2f} mm")
#                 visualizer.stl_distance_text_actor.SetInput(f"STL Distance: {dist:.2f} mm")

#         else:
#             print("Markers not found, retrying...")

#     render_window_interactor.Initialize()
#     render_window_interactor.AddObserver('TimerEvent', timer_callback)
#     render_window_interactor.CreateRepeatingTimer(16)
#     render_window_interactor.Start()







import requests
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import vtk
import pandas as pd


def to_np_point(pt):
    return np.array([pt.get('x', 0), pt.get('y', 0), pt.get('z', 0)], dtype=float)


def to_np_matrix(rot):
    return np.array([
        [rot.get('m00', 1), rot.get('m01', 0), rot.get('m02', 0)],
        [rot.get('m10', 0), rot.get('m11', 1), rot.get('m12', 0)],
        [rot.get('m20', 0), rot.get('m21', 0), rot.get('m22', 1)]
    ], dtype=float)


def fetch_marker_points(url, ref_marker_name, tool_marker_name):
    """
    Fetches from ReferenceSpace endpoint:
      GET http://localhost:8081/ReferenceSpace?data={"ReferenceMarker":"6709"}
    Response already provides positions in the reference marker's coordinate space.
    Looks for the tool marker (21007) inside the response.
    """
    try:
        import json as _json
        params = _json.dumps({"ReferenceMarker": str(ref_marker_name)})
        response = requests.get(url, params={"data": params}, timeout=5)
        response.raise_for_status()
        cam_data = response.json()
        print(f"[DEBUG] Response keys: {list(cam_data.keys()) if isinstance(cam_data, dict) else type(cam_data)}")
        if isinstance(cam_data, dict) and 'RegisteredMarkersList' in cam_data:
            names = [str(m.get('MarkerName', m.get('markerName', m.get('name', '???')))) for m in (cam_data['RegisteredMarkersList'] or [])]
            print(f"[DEBUG] RegisteredMarkersList names: {names}")

        # ── Try ReferenceSpace response format ────────────────────────────────
        # Expected: list of marker dicts OR dict with marker list
        marker_list = None
        if isinstance(cam_data, list):
            marker_list = cam_data
        elif isinstance(cam_data, dict):
            for key in ['RegisteredMarkersList', 'MarkersList', 'markers', 'Markers', 'data']:
                if key in cam_data:
                    marker_list = cam_data[key]
                    break
            if marker_list is None:
                marker_list = [cam_data]

        data_ref  = None
        data_tool = None

        for marker in marker_list:
            name = str(marker.get('MarkerName', marker.get('markerName', marker.get('name', ''))))

            if name == str(ref_marker_name):
                top = marker.get('Top', marker)
                data_ref = {
                    "rotation": top.get('rotation', {}),
                    "positions": {
                        "point":  top.get('point',  {}),
                        "point1": top.get('point1', {}),
                        "point2": top.get('point2', {}),
                        "point3": top.get('point3', {}),
                        "point4": top.get('point4', {}),
                    }
                }

            elif name == str(tool_marker_name):
                top = marker.get('Top', marker)
                data_tool = {
                    "positions": {
                        "point":  top.get('point',  {}),
                        "point1": top.get('point1', {}),
                        "point2": top.get('point2', {}),
                        "point3": top.get('point3', {}),
                        "point4": top.get('point4', {}),
                        "point5": top.get('point5', {}),
                    }
                }

        if data_ref is None or data_tool is None:
            missing = []
            if data_ref  is None: missing.append(f"ref({ref_marker_name})")
            if data_tool is None: missing.append(f"tool({tool_marker_name})")
            raise ValueError(f"Markers not found in response: {missing}")

        return data_ref, data_tool

    except Exception as e:
        print(f"Error fetching marker data: {e}")
        return None, None


class MarkerVisualizer:
    def __init__(self, renderer, render_window):
        self.renderer = renderer
        self.render_window = render_window
        self.sphere_actors = []
        # self.sphere_actor = []
        self.landmark_actors = []
        self.stl_actor = None
        self.landmarks = None
        self.implicit_distance = None
        self.bone_polydata = None
        # RMSE text actor for landmark registration
        self.rmse_text_actor = vtk.vtkTextActor()
        self.rmse_text_actor.SetInput("Landmark RMSE: N/A")
        self.rmse_text_actor.GetTextProperty().SetFontSize(36)
        self.rmse_text_actor.GetTextProperty().SetColor(0.2, 1.0, 0.2)
        self.rmse_text_actor.GetTextProperty().SetBold(1)
        self.rmse_text_actor.GetTextProperty().SetJustificationToRight()
        self.rmse_text_actor.GetTextProperty().SetVerticalJustificationToTop()
        self.renderer.AddActor(self.rmse_text_actor)
        # STL distance text actor
        self.stl_distance_text_actor = vtk.vtkTextActor()
        self.stl_distance_text_actor.SetInput("Surface Distance: N/A")
        self.stl_distance_text_actor.GetTextProperty().SetFontSize(36)
        self.stl_distance_text_actor.GetTextProperty().SetColor(0.2, 1.0, 0.2)
        self.stl_distance_text_actor.GetTextProperty().SetBold(1)
        self.stl_distance_text_actor.GetTextProperty().SetJustificationToRight()
        self.stl_distance_text_actor.GetTextProperty().SetVerticalJustificationToTop()
        self.renderer.AddActor(self.stl_distance_text_actor)

    def update_text_position(self):
        """Update text positions based on window size."""
        window_size = self.render_window.GetSize()
        # Place Landmark RMSE at top-right
        self.rmse_text_actor.SetDisplayPosition(window_size[0] - 10, window_size[1] - 40)
        # Place STL Distance directly below Landmark RMSE
        self.stl_distance_text_actor.SetDisplayPosition(window_size[0] - 10, window_size[1] - 70)
        self.renderer.GetRenderWindow().Render()

    def load_stl_actor(self, stl_path, is_bone=False):
        try:
            if stl_path.lower().endswith('.ply'):
                reader = vtk.vtkPLYReader()
            else:
                reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_path)
            reader.Update()
            stl_polydata = reader.GetOutput()
            if stl_polydata.GetNumberOfPoints() == 0:
                raise ValueError(f"File {stl_path} is empty or invalid")

            if is_bone:
                # Mesh already pre-smoothed offline — just recompute normals
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(stl_polydata)
                normals.ComputePointNormalsOn()
                normals.ComputeCellNormalsOff()
                normals.SplittingOff()
                normals.ConsistencyOn()
                normals.Update()
                self.bone_polydata = normals.GetOutput()
                self.implicit_distance = vtk.vtkImplicitPolyDataDistance()
                self.implicit_distance.SetInput(self.bone_polydata)
            else:
                self.bone_polydata = stl_polydata

            stl_mapper = vtk.vtkPolyDataMapper()
            stl_mapper.SetInputData(self.bone_polydata)
            stl_actor = vtk.vtkActor()
            stl_actor.SetMapper(stl_mapper)
            return stl_actor
        except Exception as e:
            print(f"Error loading STL file {stl_path}: {e}")
            return None

    def load_landmarks(self, excel_path):
        try:
            if excel_path.lower().endswith('.csv'):
                df = pd.read_csv(excel_path)
            else:
                df = pd.read_excel(excel_path)
            # support both lowercase (x,y,z) and uppercase (X,Y,Z) columns
            df.columns = [c.lower() for c in df.columns]
            if 'x' not in df.columns or 'y' not in df.columns or 'z' not in df.columns:
                raise ValueError("File must contain 'x', 'y', 'z' columns")
            _reg_names = ["Tibial Knee Centre","Medial Plateau","Lateral Plateau",
                          "Tibial Tuberosity","Posterior Cruciate Ligament"]
            if 'landmark' in df.columns:
                _ld = {r['landmark'].strip(): [r['x'],r['y'],r['z']] for _,r in df.iterrows()}
                self.landmarks = [_ld[n] for n in _reg_names if n in _ld]
            else:
                self.landmarks = df[['x', 'y', 'z']].dropna().values[:5].tolist()
            # Project landmarks onto bone surface before rendering
            for actor in self.landmark_actors:
                self.renderer.RemoveActor(actor)
            self.landmark_actors.clear()
            if self.bone_polydata is not None:
                _loc = vtk.vtkCellLocator()
                _loc.SetDataSet(self.bone_polydata)
                _loc.BuildLocator()
                _cid = vtk.reference(0); _sid = vtk.reference(0); _d2 = vtk.reference(0.0)
            projected = []
            for lm in self.landmarks:
                if self.bone_polydata is not None:
                    closest = [0.0, 0.0, 0.0]
                    _loc.FindClosestPoint(lm, closest, _cid, _sid, _d2)
                    # push 1.5mm outward along gradient
                    grad = [0.0, 0.0, 0.0]
                    self.implicit_distance.FunctionGradient(closest, grad)
                    gn = np.array(grad); mag = np.linalg.norm(gn)
                    if mag > 1e-9:
                        gn /= mag
                        closest = [closest[0]+gn[0]*1.5, closest[1]+gn[1]*1.5, closest[2]+gn[2]*1.5]
                    projected.append(closest)
                else:
                    projected.append(lm)
            _lm_colors = [
                (0.2, 0.4, 1.0),   # TKC — blue
                (0.2, 0.85, 0.2),  # Medial Plateau — green
                (1.0, 0.2, 0.2),   # Lateral Plateau — red
                (1.0, 0.85, 0.0),  # Tibial Tuberosity — yellow
                (0.8, 0.2, 1.0),   # PCL — purple
            ]
            for i, pt in enumerate(projected):
                col = _lm_colors[i % len(_lm_colors)]
                actor = self.create_sphere_actor(pt, radius=3.0, color=col)
                self.renderer.AddActor(actor)
                self.landmark_actors.append(actor)
            self.renderer.ResetCameraClippingRange()
        except Exception as e:
            print(f"Error loading landmarks from {excel_path}: {e}")
            self.landmarks = None

    def update_landmarks_visualization(self, needle_pos, stl_distance=None):
        # Only update STL Distance — RMSE is set once at registration and kept frozen
        if stl_distance is not None:
            self.stl_distance_text_actor.SetInput(f"Surface Distance: {stl_distance:.2f} mm")
            self.stl_distance_text_actor.SetVisibility(1)       
    
    def update_tool_stl(self, transformed_tool_markers, stl_path, tool_tip_index=0):
        if len(transformed_tool_markers) < 6:
            print("Insufficient tool markers for STL update")
            return

        if self.stl_actor is None:
            self.stl_actor = self.load_stl_actor(stl_path)
            if self.stl_actor is None:
                print("Failed to load needle STL")
                return
            self.renderer.AddActor(self.stl_actor)

        markers = transformed_tool_markers
        start_point = markers[tool_tip_index]
        centroid = np.mean(markers[1:6], axis=0)
        direction_vector = centroid - start_point
        direction_vector /= np.linalg.norm(direction_vector)

        z_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(z_axis, direction_vector)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_axis = np.array([1, 0, 0])
        else:
            rotation_axis /= np.linalg.norm(rotation_axis)

        dot_product = np.dot(z_axis, direction_vector)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_degrees = np.degrees(angle)

        if angle_degrees <= 30:
            color = (0, 1, 0)
        elif angle_degrees <= 60:
            color = (1, 1, 0)
        else:
            color = (1, 0, 0)

        transform = vtk.vtkTransform()
        transform.Translate(*start_point)
        transform.RotateWXYZ(angle_degrees, *rotation_axis)
        self.stl_actor.SetUserTransform(transform)
        self.stl_actor.GetProperty().SetColor(*color)
        self.renderer.GetRenderWindow().Render()

        # Calculate STL distance for update_landmarks_visualization
        stl_distance = None
        if self.implicit_distance:
            tool_tip = transformed_tool_markers[tool_tip_index]
            dist = abs(self.implicit_distance.EvaluateFunction(tool_tip))
            if dist > 1000:
                print("Warning: Unreasonably large distance to STL surface")
                dist = float('inf')
            else:
                stl_distance = dist
            print(f"Distance from tool tip to STL surface: {dist:.2f} mm")

        self.update_landmarks_visualization(transformed_tool_markers[tool_tip_index], stl_distance)

    def update_spheres(self, positions):
        for actor in self.sphere_actors:
            self.renderer.RemoveActor(actor)
        self.sphere_actors.clear()
        for position in positions:
            actor = self.create_sphere_actor(position)
            self.renderer.AddActor(actor)
            self.sphere_actors.append(actor)
        self.renderer.GetRenderWindow().Render()

    @staticmethod
    def create_sphere_actor(position, radius=10.0, color=(1, 1, 1)):
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(position[0], position[1], position[2])
        sphere_source.SetRadius(radius)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(*color)
        return sphere_actor

# ─────────────────────────────────────────────────────────────────────────────
# Procrustes: tracker space → phantom space  (same logic as phantom_infer.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_procrustes(src, tgt):
    """
    Rigid registration only (scale=1.0) — both spaces are in mm.
    T(x) = R @ x + t
    """
    mu_s  = src.mean(0);  mu_t = tgt.mean(0)
    sc    = src - mu_s;   tc   = tgt - mu_t
    H     = sc.T @ tc
    U, S, Vt = np.linalg.svd(H)
    d    = np.sign(np.linalg.det(Vt.T @ U.T))
    D    = np.diag([1., 1., d])
    Rmat = (Vt.T @ D @ U.T).astype(np.float32)
    t    = mu_t - Rmat @ mu_s
    return Rmat, t.astype(np.float32), 1.0


def apply_tracker_to_phantom(pts, Rmat, t, scale):
    """Convert tracker-space points → phantom space."""
    pts = np.array(pts, dtype=np.float32)
    return (scale * (Rmat @ pts.T).T + t).astype(np.float32)


if __name__ == "__main__":

    # ── Phantom landmarks (target space) ─────────────────────────────────────
    import pandas as _pd
    landmarks_path = r"C:\Users\hticl\Downloads\Windows_Tracking_Code\Ragavan_Tibia\phantom_19_landmrks.csv"
    _df = _pd.read_csv(landmarks_path)
    _df.columns = [c.lower() for c in _df.columns]

    # 5 registration landmarks — selected by name from CSV
    LM_NAMES_5 = [
        "Tibial Knee Centre",
        "Medial Plateau",
        "Lateral Plateau",
        "Tibial Tuberosity",
        "Posterior Cruciate Ligament",
    ]
    N_LM_TOTAL = 5

    # Build PHANTOM_LM by matching landmark names
    _lm_dict = {}
    if 'landmark' in _df.columns:
        for _, row in _df.iterrows():
            _lm_dict[row['landmark'].strip()] = [row['x'], row['y'], row['z']]
    PHANTOM_LM = np.array([_lm_dict[n] for n in LM_NAMES_5], dtype=np.float32)

    print(f"Phantom landmarks (TARGET space) - 3 registration points:")
    for i,(n,p) in enumerate(zip(LM_NAMES_5, PHANTOM_LM)):
        print(f"  [{i+1}] {n:<35} {p.round(1)}")

    # ── Registration state ────────────────────────────────────────────────────
    registered     = [False]
    _reg    = [None, None, None]   # [R, t, scale] — mutable so closures share it
    _warped = [None]               # warped landmark positions after registration

    # ── VTK setup ─────────────────────────────────────────────────────────────
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1280, 900)
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    visualizer = MarkerVisualizer(renderer, render_window)

    # Load bone
    stl_path = r"C:\Users\hticl\Downloads\Windows_Tracking_Code\Ragavan_Tibia\output_Tibia\infer_output\19_phantom_smooth.ply"
    stl_actor = visualizer.load_stl_actor(stl_path, is_bone=True)

    # Pre-load bone as open3d target for ICP
    import open3d as _o3d
    _bone_mesh = _o3d.io.read_triangle_mesh(stl_path)
    _bone_mesh.compute_vertex_normals()
    _bone_target = _bone_mesh.sample_points_uniformly(8000)
    _bone_target.estimate_normals()
    _bone_arr = np.asarray(_bone_target.points, dtype=np.float64)
    print(f"ICP target: {len(_bone_arr)} surface points loaded")
    if stl_actor is None:
        print("Failed to load bone, exiting."); exit(1)
    stl_actor.GetProperty().SetColor(0.38, 0.38, 0.38)   # darker grey = more contrast
    stl_actor.GetProperty().SetAmbient(0.12)              # less ambient = darker shadows
    stl_actor.GetProperty().SetDiffuse(0.80)
    stl_actor.GetProperty().SetSpecular(0.55)             # stronger specular highlight
    stl_actor.GetProperty().SetSpecularPower(80)          # tighter highlight = sharper
    stl_actor.GetProperty().SetInterpolationToPhong()
    renderer.AddActor(stl_actor)
    renderer.SetBackground(0.0, 0.0, 0.0)
    # Add key light + fill light for smooth look
    key_light = vtk.vtkLight()
    key_light.SetLightTypeToSceneLight()
    key_light.SetPosition(200, 400, 500)
    key_light.SetFocalPoint(150, 50, 1835)
    key_light.SetIntensity(1.0)
    key_light.SetColor(1.0, 1.0, 1.0)
    renderer.AddLight(key_light)
    fill_light = vtk.vtkLight()
    fill_light.SetLightTypeToSceneLight()
    fill_light.SetPosition(-200, -100, 300)
    fill_light.SetFocalPoint(150, 50, 1835)
    fill_light.SetIntensity(0.4)
    fill_light.SetColor(0.8, 0.8, 1.0)
    renderer.AddLight(fill_light)

    # Axes
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(50, 50, 50)
    axes.SetShaftTypeToLine()
    renderer.AddActor(axes)

    # Load and display boundary points projected onto bone surface
    import open3d as _o3d_bp
    _bp = _o3d_bp.io.read_point_cloud(
        r"C:\Users\hticl\Downloads\phantom_20_boundary.ply")
    _bp_pts = np.asarray(_bp.points)
    # Push boundary points 2.5mm outward along bone surface normal
    _bp_vtk = vtk.vtkPoints()
    if visualizer.bone_polydata is not None and visualizer.implicit_distance is not None:
        _bp_loc = vtk.vtkCellLocator()
        _bp_loc.SetDataSet(visualizer.bone_polydata)
        _bp_loc.BuildLocator()
        _cid_ = vtk.reference(0); _sid_ = vtk.reference(0); _d2_ = vtk.reference(0.0)
        for _pt in _bp_pts:
            _cl = [0.0, 0.0, 0.0]
            _bp_loc.FindClosestPoint(_pt.tolist(), _cl, _cid_, _sid_, _d2_)
            _bp_vtk.InsertNextPoint(_cl[0], _cl[1], _cl[2])
    else:
        for _pt in _bp_pts:
            _bp_vtk.InsertNextPoint(_pt[0], _pt[1], _pt[2])
    _bp_pd = vtk.vtkPolyData()
    _bp_pd.SetPoints(_bp_vtk)
    _bp_glyph = vtk.vtkVertexGlyphFilter()
    _bp_glyph.SetInputData(_bp_pd)
    _bp_glyph.Update()
    _bp_mapper = vtk.vtkPolyDataMapper()
    _bp_mapper.SetInputConnection(_bp_glyph.GetOutputPort())
    _bp_actor = vtk.vtkActor()
    _bp_actor.SetMapper(_bp_mapper)
    _bp_actor.GetProperty().SetColor(0.1, 0.5, 1.0)   # blue
    _bp_actor.GetProperty().SetPointSize(7)
    _bp_actor.GetProperty().SetRenderPointsAsSpheres(True)
    renderer.AddActor(_bp_actor)
    _bp_actor.SetVisibility(1)   # show immediately
    print(f"Boundary points: {len(_bp_pts)}")

    # Load landmarks for visualisation
    visualizer.load_landmarks(landmarks_path)
    if visualizer.landmarks is None:
        print("Failed to load landmarks, exiting."); exit(1)

    # Single big RMSE label on screen
    reg_text = vtk.vtkTextActor()
    reg_text.SetInput(f"Touch Tibial Knee Centre then SPACE  (0/5)")
    reg_text.GetTextProperty().SetFontSize(28)
    reg_text.GetTextProperty().SetColor(1, 1, 0)
    reg_text.SetDisplayPosition(10, 30)
    renderer.AddActor(reg_text)

    # Focus camera on the bone — zoom to fill window
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    bounds = stl_actor.GetBounds()  # xmin,xmax,ymin,ymax,zmin,zmax
    cx = (bounds[0]+bounds[1])/2
    cy = (bounds[2]+bounds[3])/2
    cz = (bounds[4]+bounds[5])/2
    diag = ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)**0.5
    # Point camera at tibial plateau (proximal = high Z end)
    focal_z = bounds[5] - (bounds[5]-bounds[4]) * 0.25   # 25% from top (knee end)
    cam.SetFocalPoint(cx, cy, focal_z)
    cam.SetPosition(cx + diag*0.2, cy - diag*1.0, focal_z + diag*0.5)
    cam.SetViewUp(0, 0, 1)
    cam.SetViewAngle(30)
    renderer.ResetCameraClippingRange()
    render_window.Render()
    visualizer.update_text_position()

    render_window_interactor.AddObserver(
        'ModifiedEvent', lambda o, e: visualizer.update_text_position())

    url            = 'http://localhost:8081/ReferenceSpace'
    ref_marker_name  = "6709"
    tool_marker_name = "21007"
    tool_tip_index   = 0

    def run_icp(tracker_pts):
        """
        Fit tracker surface points to bone mesh using PCA init + ICP.
        Returns 4x4 transform (tracker → phantom) and surface RMSE.
        """
        src_arr = np.array(tracker_pts, dtype=np.float64)
        src_pcd = _o3d.geometry.PointCloud()
        src_pcd.points = _o3d.utility.Vector3dVector(src_arr)

        src_mean = src_arr.mean(0)
        tgt_mean = _bone_arr.mean(0)

        _, _, Vs = np.linalg.svd(src_arr - src_mean, full_matrices=False)
        _, _, Vt = np.linalg.svd(_bone_arr - tgt_mean, full_matrices=False)

        best_rmse = np.inf
        best_T4   = np.eye(4)

        for s0 in (1, -1):
          for s1 in (1, -1):
            R_pca = np.eye(3, dtype=np.float64)
            R_pca = (Vt * np.array([s0, s1, 1.0])[:, None]).T @ Vs
            t_pca = tgt_mean - R_pca @ src_mean
            T_init = np.eye(4)
            T_init[:3, :3] = R_pca
            T_init[:3,  3] = t_pca

            res = _o3d.pipelines.registration.registration_icp(
                src_pcd, _bone_target, 30.0, T_init,
                _o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                _o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))

            if res.inlier_rmse > 0 and res.inlier_rmse < best_rmse:
                best_rmse = res.inlier_rmse
                best_T4   = res.transformation

        return best_T4, best_rmse

    def apply_T4(pts, T4):
        pts = np.array(pts, dtype=np.float64)
        h   = np.hstack([pts, np.ones((len(pts), 1))])
        return (T4 @ h.T).T[:, :3].astype(np.float32)

    # ── State ────────────────────────────────────────────────────────────────────
    N_LM_TOTAL    = 5           # 5-landmark registration, no Phase 1 warm-up
    LM_NAMES_3    = LM_NAMES_5  # alias kept for compatibility

    phase         = [2]         # start directly at registration (no Phase 1)
    blue_tracker  = []
    blue_actors   = []
    reg_pts       = []          # registration touch points (tracker space)
    reg_actors    = []          # blue dots after registration
    last_tip      = [None]

    # Live needle tip sphere (moves in tracker space during phase 1 & 2)
    _tip_src = vtk.vtkSphereSource()
    _tip_src.SetRadius(2.0)
    _tip_src.SetPhiResolution(10); _tip_src.SetThetaResolution(10)
    _tip_mapper = vtk.vtkPolyDataMapper()
    _tip_mapper.SetInputConnection(_tip_src.GetOutputPort())
    tip_live_actor = vtk.vtkActor()
    tip_live_actor.SetMapper(_tip_mapper)
    tip_live_actor.GetProperty().SetColor(1, 1, 0)   # yellow live dot
    tip_live_actor.SetVisibility(0)
    renderer.AddActor(tip_live_actor)

    # Live position text
    tip_pos_text = vtk.vtkTextActor()
    tip_pos_text.GetTextProperty().SetFontSize(20)
    tip_pos_text.GetTextProperty().SetColor(1, 1, 0)
    tip_pos_text.SetDisplayPosition(10, 60)
    renderer.AddActor(tip_pos_text)

    # Per-landmark distance display (shown in phase 3)
    lm_dist_text = vtk.vtkTextActor()
    lm_dist_text.GetTextProperty().SetFontSize(22)
    lm_dist_text.GetTextProperty().SetColor(0.9, 0.9, 0.9)
    lm_dist_text.GetTextProperty().SetFontFamilyToCourier()
    lm_dist_text.SetDisplayPosition(10, 120)
    lm_dist_text.SetVisibility(0)
    renderer.AddActor(lm_dist_text)

    paint_actors    = []
    paint_colors    = [(0.1, 0.5, 1.0)]   # blue only
    paint_color_idx = [0]
    last_paint_pos  = [None]
    PAINT_MIN_DIST  = 5.0

    def _get_cam2dd(data_ref):
        rot = np.array([data_ref['rotation'].get('x', 0.0),
                        data_ref['rotation'].get('y', 0.0),
                        data_ref['rotation'].get('z', 0.0),
                        data_ref['rotation'].get('w', 1.0)])
        R_mat  = R.from_quat(rot).as_matrix()
        t_orig = np.array([list(data_ref['positions'].values())[0].get('x', 0),
                           list(data_ref['positions'].values())[0].get('y', 0),
                           list(data_ref['positions'].values())[0].get('z', 0)])
        cam2dd = np.eye(4)
        cam2dd[:3, :3] = R_mat.T
        cam2dd[:3,  3] = -R_mat.T @ t_orig
        return cam2dd

    def _tp(d, cam2dd):
        pos = np.array([d.get('x',0), d.get('y',0), d.get('z',0)])
        return np.dot(cam2dd, np.append(pos, 1))[:3]

    def timer_callback(obj, event):
        data_ref, data_tool = fetch_marker_points(url, ref_marker_name, tool_marker_name)
        if not (data_ref and data_tool):
            return

        cam2dd   = _get_cam2dd(data_ref)
        all_ref  = [_tp(v, cam2dd) for v in data_ref['positions'].values()]
        all_tool = [_tp(v, cam2dd) for v in data_tool['positions'].values()]
        last_tip[0] = np.array(all_tool[tool_tip_index])
        tip = last_tip[0]

        if phase[0] == 2:
            # Show needle STL live in tracker space so user sees where they're touching
            visualizer.update_tool_stl(all_tool,
                r"C:\Users\hticl\Downloads\Needle_Stl.stl", tool_tip_index)
            _tip_src.SetCenter(tip[0], tip[1], tip[2])
            _tip_src.Update()
            tip_live_actor.SetVisibility(1)
            n = len(reg_pts)
            name = LM_NAMES_3[n] if n < N_LM_TOTAL else "done"
            reg_text.SetInput(f"Touch {name} then SPACE  ({n}/{N_LM_TOTAL})")
            reg_text.GetTextProperty().SetColor(1, 1, 0)
            tip_pos_text.SetInput(f"Tip: ({tip[0]:.1f}, {tip[1]:.1f}, {tip[2]:.1f})")

        elif phase[0] == 3:
            tip_live_actor.SetVisibility(0)
            tip_pos_text.SetInput("")
            if _reg[0] is None:
                return
            all_ref_ph  = [apply_tracker_to_phantom(p, _reg[0], _reg[1], _reg[2]) for p in all_ref]
            all_tool_ph = [apply_tracker_to_phantom(p, _reg[0], _reg[1], _reg[2]) for p in all_tool]
            visualizer.update_tool_stl(all_tool_ph,
                r"C:\Users\hticl\Downloads\Needle_Stl.stl", tool_tip_index)

            tip_ph = np.array(all_tool_ph[tool_tip_index], dtype=np.float64)

            # Live RMSE = distance to nearest touched landmark (warped positions)
            if _warped[0] is not None:
                dists_to_lm = [float(np.linalg.norm(tip_ph - lm)) for lm in _warped[0]]
                nearest_idx = int(np.argmin(dists_to_lm))
                live_rmse   = dists_to_lm[nearest_idx]
                r_col = (0,1,0) if live_rmse < 3 else (1,0.6,0) if live_rmse < 8 else (1,0,0)
                visualizer.rmse_text_actor.GetTextProperty().SetColor(*r_col)
                visualizer.rmse_text_actor.SetInput(f"Landmark RMSE: {live_rmse:.2f} mm")
            lm_dist_text.SetVisibility(0)

            if visualizer.implicit_distance:
                dist  = abs(visualizer.implicit_distance.EvaluateFunction(
                    [float(tip_ph[0]), float(tip_ph[1]), float(tip_ph[2])]))
                color = (0,1,0) if dist < 3 else (1,0.6,0) if dist < 8 else (1,0,0)
                visualizer.stl_distance_text_actor.GetTextProperty().SetColor(*color)
                visualizer.stl_distance_text_actor.SetInput(f"Surface Distance: {dist:.2f} mm")
                reg_text.GetTextProperty().SetColor(*color)
                reg_text.SetInput(f"Surface Distance: {dist:.2f} mm")

                if dist < 10.0:
                    if last_paint_pos[0] is None or \
                       np.linalg.norm(tip_ph - last_paint_pos[0]) > PAINT_MIN_DIST:
                        c = paint_colors[paint_color_idx[0] % len(paint_colors)]
                        dot = visualizer.create_sphere_actor(tip_ph, radius=1.5, color=c)
                        renderer.AddActor(dot)
                        paint_actors.append(dot)
                        last_paint_pos[0] = tip_ph.copy()
                        if len(paint_actors) % 20 == 0:
                            paint_color_idx[0] += 1

        render_window.Render()

    def keypress_callback(obj, event):
        key = obj.GetKeySym()

        if key == 'space':
            tip = last_tip[0]
            if tip is None:
                return

            # ── Registration: collect 5 landmarks ────────────────────────────
            if phase[0] == 2:
                if len(reg_pts) >= N_LM_TOTAL:
                    return
                reg_pts.append(tip.copy())
                n = len(reg_pts)
                print(f"  [Reg {n}] {LM_NAMES_3[n-1]}  tracker={tip.round(2)}")
                if n == N_LM_TOTAL:
                    src = np.array(reg_pts, dtype=np.float32)
                    tgt = PHANTOM_LM
                    _reg[0], _reg[1], _reg[2] = compute_procrustes(src, tgt)
                    warped = apply_tracker_to_phantom(src, _reg[0], _reg[1], _reg[2])
                    _warped[0] = warped
                    res = np.linalg.norm(warped - tgt, axis=1)
                    rmse = res.mean()
                    print(f"  Registration done. mean={rmse:.2f}mm  max={res.max():.2f}mm")
                    for nm, r in zip(LM_NAMES_3, res):
                        print(f"    {nm:<35} {r:.2f} mm")
                    registered[0] = True
                    phase[0] = 3

                    # Keep existing colored landmark dots — do not remove them
                    for a in reg_actors:
                        renderer.RemoveActor(a)
                    reg_actors.clear()

                    visualizer.rmse_text_actor.SetInput(f"Landmark RMSE: {rmse:.2f} mm")
                    visualizer.stl_distance_text_actor.SetInput("Surface Distance: -- mm")

            render_window.Render()

        elif key == 'p':
            if last_tip[0] is not None:
                tip = last_tip[0]
                print(f"[P] Tracker tip: ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})")
                if registered[0] and _reg[0] is not None:
                    ph = apply_tracker_to_phantom(tip, _reg[0], _reg[1], _reg[2])
                    print(f"[P] Phantom tip: ({ph[0]:.3f}, {ph[1]:.3f}, {ph[2]:.3f})")

        elif key == 'r':
            phase[0] = 2
            registered[0] = False
            _reg[0] = _reg[1] = _reg[2] = None
            _warped[0] = None
            reg_pts.clear()
            for a in reg_actors:   renderer.RemoveActor(a)
            reg_actors.clear()
            for a in paint_actors: renderer.RemoveActor(a)
            paint_actors.clear()
            last_paint_pos[0] = None
            paint_color_idx[0] = 0
            visualizer.rmse_text_actor.SetInput("Landmark RMSE: N/A")
            visualizer.stl_distance_text_actor.SetInput("STL Distance: N/A")
            reg_text.SetInput(f"Touch Tibial Knee Centre then SPACE  (0/3)")
            reg_text.GetTextProperty().SetColor(1, 1, 0)
            render_window.Render()
            print("Reset — touch 5 landmarks again.")

    render_window_interactor.Initialize()
    render_window_interactor.AddObserver('TimerEvent',   timer_callback)
    render_window_interactor.AddObserver('KeyPressEvent', keypress_callback)
    render_window_interactor.CreateRepeatingTimer(16)
    render_window_interactor.Start()