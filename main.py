from preprocess import Preprocessing
from slam import Slam

def main():
    # All data in this path
    path = 'mav0'
    preprocessor = Preprocessing()
    # We need to process all of this data, and return an object that we can easily pass into Slam()
    # Since we don't need real time performance, just batch it all at once.
    data = preprocessor.get_data(path)
    
    slam = Slam()
    slam.generate_map(data)
    map = slam.get_map() 



if __name__ == main:
    main()
