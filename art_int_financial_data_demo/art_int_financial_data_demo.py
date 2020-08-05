"""
DOCSTRING
"""
import flask
import json
import logging
import os
import pandas
import sklearn
import tempfile
import unittest

PARENT_DIR_PATH = os.path.dirname(os.path.realpath(os.path.join(__file__, '..')))
DATASET_FILENAME = os.path.join(PARENT_DIR_PATH, "data", "creditcard.csv")
MODEL_FILENAME = os.path.join(PARENT_DIR_PATH, "models", "model.pickle")

class EndpointTest(unittest.TestCase):
    """
    Flask app tests
    """

    def post(self, data):
        """
        DOCSTRING
        """
        return self.client.post(
            path=self.endpoint, data=json.dumps(data), content_type='application/json')

    def setUp(self):
        """
        DOCSTRING
        """
        FlaskApp().app.config['DATABASE'] = tempfile.mkstemp()
        app.testing = True
        self.client = app.test_client()
        self.endpoint = "/predict"

    def tearDown(self):
        """
        DOCSTRING
        """
        pass

    def test_empty_body(self):
        """
        DOCSTRING
        """
        with app.app_context():
            response = self.post({})
            self.assertEqual(response.status_code, 400)
            response_object = json.loads(response.data)
            self.assertTrue("error" in response_object)

    def test_multiple_data_points(self):
        """
        DOCSTRING
        """
        features = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        with app.app_context():
            response = self.post(dict(features=features))
            self.assertEqual(response.status_code, 200)
            response_object = json.loads(response.data)
            self.assertTrue("scores" in response_object)
            self.assertEqual(len(response_object["scores"]), 2)
            self.assertTrue(isinstance(response_object["scores"][0], float))
            self.assertTrue(isinstance(response_object["scores"][1], float))

    def test_no_features(self):
        """
        DOCSTRING
        """
        with app.app_context():
            response = self.post(dict(features=None))
            self.assertEqual(response.status_code, 400)
            response_object = json.loads(response.data)
            self.assertTrue("error" in response_object)

    def test_single_data_point(self):
        """
        DOCSTRING
        """
        features = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with app.app_context():
            response = self.post(dict(features=features))
            self.assertEqual(response.status_code, 200)
            response_object = json.loads(response.data)
            self.assertTrue("scores" in response_object)
            self.assertEqual(len(response_object["scores"]), 1)
            self.assertTrue(isinstance(response_object["scores"][0], float))

class FlaskApp:
    """
    DOCSTRING
    """

    def __init__(self):
        app = flask.Flask("Fraud Detection")
        app.model = sklearn.externals.joblib.load(self.MODEL_FILENAME)

    @app.route(u"/predict", methods=[u"POST"])
    def predict_fraud():
        input_data = flask.request.get_json()
        if u"features" not in input_data:
            return json.dumps({u"error": u"No features found in input"}), 400
        if not input_data[u"features"] or not isinstance(input_data[u"features"], list):
            return json.dumps({u"error": u"No feature values available"}), 400
        if isinstance(input_data[u"features"][0], list):
            results = app.model.predict_proba(input_data[u"features"]).tolist()
        else:
            results = app.model.predict_proba([input_data[u"features"]]).tolist()
        return json.dumps({u"scores": [result[1] for result in results]}), 200

class Train:
    """
    DOCSTRING
    """

    logger = logging.getLogger('training')

    def __call__(self):
        """
        Find the best model to fit the dataset and save it into file
        """
        grid_search = new_grid_search()
        run_grid_search(grid_search)
        save_search_results(grid_search)

    def new_grid_search(self):
        """
        Create new GridSearch obj with models pipeline
        """
        pipeline = sklearn.pipeline.Pipeline(
            [(u"clf", sklearn.linear_model.logistic.LogisticRegression(class_weight="balanced"))])
        search_params = {"clf__C": (1e-4, 1e-2, 1e0, 1e2, 1e4)}
        return sklearn.model_selection.GridSearchCV(
            estimator=pipeline,
            param_grid=search_params,
            scoring="recall_macro",
            cv=10,
            n_jobs=-1,
            verbose=3)

    def run_grid_search(self, grid_search, show_evaluation=True):
        """
        Run the GridSearch algorithm and compute evaluation metrics
        """
        X_train, X_test, y_train, y_test = split_dataset()
        grid_search.fit(X_train, y_train)
        #for key, value in grid_search.cv_results_.items():
        #    print key, value
        predictions = grid_search.predict(X_test)
        if show_evaluation:
            logger.debug(
                "macro_recall: %s",
                sklearn.metrics.classification.recall_score(
                    y_test, predictions, average="macro"))
            logger.debug(
                sklearn.metrics.classification.precision_recall_fscore_support(
                    y_test, predictions))
            logger.debug(
                sklearn.metrics.classification.confusion_matrix(y_test, predictions))

    def save_search_results(self, grid_search):
        """
        Serialize model into file
        """
        sklearn.externals.joblib.dump(grid_search.best_estimator_, MODEL_FILENAME)
        #clf = sklearn.externals.joblib.load(model_dump_filename)

    def split_dataset(self):
        """
        Read and split dataset into train and test subsets
        """
        df = pandas.read_csv(DATASET_FILENAME, header=0)
        X = df[df.columns[:-1]].as_matrix()
        y = df[df.columns[-1]].as_matrix()
        return sklearn.model_selection.train_test_split(
            X, y, test_size=0.2, random_state=42)

if __name__ == '__main__':
    endpoint_test = EndpointTest()
    endpoint_test()
