import mylib
import logging
import etl.extract as extract
import etl.transform as transform
import etl.load as load
import analysis.model as model
import analysis.evaluate as eval
import vis.visualizations as vis
logger = logging.getLogger(__name__)

def main():
        logging.basicConfig(filename='alex.log', level=logging.INFO)
        logger.info('started creating raw CSVs for pm25 and temperature')
        extract.air_pollutant()
        extract.weather()
        
        
        logger.info('Finished creating raw CSVs')
        
        
        
        logger.info('starting merge')
        extract.merge()
        logger.info('finished merge. pollutant_merge.csv created.')
        
        
        #TRANSFORMING SECTION
        
        logger.info("dropping columns")
        transform.drop_columns()
        logger.info("Columns dropped")

        logger.info("getting rid of outliers")
        transform.outliers()
        logger.info("outliers removed.")

        #splitting into testing and training datasets
        logger.info("attempting to split dataset")
        X, y = transform.split_into_train_and_test()
        X_train, X_test, y_train, y_test = transform.split_train_test_data(X, y)
        logger.info("datasets split into training and testing sets.")

        #normalizing both datasets
        logger.info("attempting to normalize")
        train_X_scaled, test_X_scaled, scaler = transform.normalize_features(X_train, X_test)
        logger.info("normalized successfully.")

        #creating the model
        logger.info("creating linear regression")
        lr_model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred = model.train_lin_reg(train_X_scaled, y_train, test_X_scaled, y_test)
        logger.info("model trained")



        #LOADING

        logger.info("attempting to load data")
        load.load_data(X_train_scaled, X_test_scaled, y_train, y_test, y_pred, lr_model)
        logger.info("finished loading data.")


        #EVALUATING
        logger.info("evaluating model and creating metrics")
        eval.evaluate_model(X_test_scaled, y_test)
        logger.info("finished creating metrics")


        #VISUALIZATION
        logger.info("Creating plot(s)")
        vis.create_linreg_Scatter(y_test,y_pred)
        logger.info("visualization created successfully.")









        


        # transform.outliers()
        # transform.split_into_train_and_test()
        # model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred = transform.train_lin_reg()

        # load.load_data(X_train_scaled, X_test_scaled, y_train, y_test)
        


        





if __name__ == "__main__":
        main()