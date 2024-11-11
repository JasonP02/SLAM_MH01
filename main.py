from slam import Slam
from utils.visualization import visualize_map        

def main():
    data_path = r"C:\Users\jason\Desktop\mav0"  
    slam = Slam(data_path=data_path)
    
    slam.run()
    map = slam.get_map()

    # You can add visualization or further processing here if needed
    visualize_map(map)

if __name__ == "__main__":
    main()
