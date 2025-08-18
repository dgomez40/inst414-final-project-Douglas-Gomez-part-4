
import etl.extract as extract
import etl.transform as transform
import etl.load as load

def main():
        # extract.air_pollutant()
        # extract.weather()
        extract.merge()
        
        transform.drop_columns()
        transform.outliers()
        transform.split_into_train_and_test()
        model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred = transform.train_lin_reg()

        load.load_data(X_train_scaled, X_test_scaled, y_train, y_test)
        


        





if __name__ == "__main__":
        main()