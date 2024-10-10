from utils import Map

class Slam:
    def __init__(self, robot_data) -> None:
        self.map = Map()  # create an instance of Map
        self.data = robot_data  # store robot data in self.data
        
    def process_frames(self):
        pass  # Implement frame processing logic here

    def get_map(self):
        return self.map  # Return the map object

    def generate_map(self):
        pass  # Implement map generation logic here