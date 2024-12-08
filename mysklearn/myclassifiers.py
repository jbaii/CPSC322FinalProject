from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import numpy as np
from scipy.stats import mode

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_reg = self.regressor.predict(X_test)
        y_cls = []
        for y_val in y_reg:
            y_cls.append(self.discretizer(y_val))
        return y_cls

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        for test_instance in X_test:
            dists = []
            for train_instance in self.X_train:
                dist = myutils.compute_Euclidean_distance(test_instance, train_instance)
                dists.append(dist)
            sorted_indices = sorted(range(len(dists)), key=lambda i: dists[i])
            sorted_dists = [dists[i] for i in sorted_indices]
            distances.append(sorted_dists[:self.n_neighbors])
            neighbor_indices.append(sorted_indices[:self.n_neighbors])
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        distances, neighbor_indices = self.kneighbors(X_test)
        for i in range(len(X_test)):
            neighbors = neighbor_indices[i]
            neighbor_labels = [self.y_train[j] for j in neighbors]
            vote_counts = {}
            for label in neighbor_labels:
                if label not in vote_counts:
                    vote_counts[label] = 1
                else:
                    vote_counts[label] += 1
            #print(vote_counts)
            # Get most frequent label as prediction
            y_predicted.append(max(vote_counts, key=vote_counts.get))
            #y_predicted.append(most_common_label)
        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self, strategy="most_frequent"):
        """Initializer for DummyClassifier.

        """
        self.strategy = strategy
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        #most_comomn_label = 
        y_label_dict = {}
        for y_label in y_train:
            if y_label not in y_label_dict:
                y_label_dict[y_label] = 1
            else:
                y_label_dict[y_label] += 1
        for label, val in y_label_dict.items():
	        y_label_dict[label] = val / len(y_train)
        self.most_common_label = max(y_label_dict, key=y_label_dict.get)
        if self.strategy == "stratified":
            y_label_key = list(y_label_dict.keys())
            ylabel_val = [y_label_dict[key] for key in y_label_key]
            y_label_val_cul= [sum(ylabel_val[:i+1]) for i in range(len(y_label_key))]
            self.y_label_key = y_label_key
            self.y_label_val_cul = y_label_val_cul

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.strategy == "most_frequent":
            y_predicted = [self.most_common_label for _ in range(len(X_test))]
            return y_predicted
        elif self.strategy == "stratified":
            y_predicted = []
            for i in range(len(X_test)):
                random_num = np.random.uniform(0,1)
                for idx,cul_val in enumerate(self.y_label_val_cul):
                    if random_num < cul_val:
                        y_predicted.append(self.y_label_key[idx])
                        break
            return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {}
        self.posteriors = {}

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        feature_counts = {}
        class_counts = {}

        # Count occurrences of each class
        for label in y_train:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        for label in class_counts:
            self.priors[label] = class_counts[label] / len(y_train)

    # Initialize the dictionary to count occurrences of feature values per class
        for label in class_counts:
            if label not in feature_counts:
                feature_counts[label] = [{} for _ in range(len(X_train[0]))]

    # Count occurrences of feature values per class
        for features, label in zip(X_train, y_train):
            for idx, feature in enumerate(features):
                if feature not in feature_counts[label][idx]:
                    feature_counts[label][idx][feature] = 0
                feature_counts[label][idx][feature] += 1

    # Calculate probabilities of each feature value given the class
        self.posteriors = {}
        for cls, counts in feature_counts.items():
            self.posteriors[cls] = []
            for idx in range(len(X_train[0])):
                total = sum(counts[idx].values())
                probs = {feature: (count / total) for feature, count in counts[idx].items()}
                self.posteriors[cls].append(probs)
            


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        # Calculate log probabilities for each class to prevent underflow
        for features in X_test:
            log_probs = {}
            for cls, class_posts in self.posteriors.items():
                prior = self.priors[cls]
                log_prob = np.log(prior)
                for idx, feature in enumerate(features):
                    if feature in class_posts[idx]:
                        log_prob += np.log(class_posts[idx][feature])
                    else:
                        log_prob += np.log(1e-6)  # Apply Laplace smoothing for unseen features
                log_probs[cls] = log_prob

            # Choose the class with the highest probability
            predicted_class = max(log_probs, key=log_probs.get)
            y_predicted.append(predicted_class)
        return y_predicted


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        available_attributes = list(range(len(X_train[0])))
        self.tree = self._tdidt(X_train, y_train, available_attributes)
    def _tdidt(self, X, y, available_attributes, parent_sample_count=None):
        if parent_sample_count is None:
            parent_sample_count = len(y)
        if len(set(y)) == 1:
            return ["Leaf", y[0], len(y), parent_sample_count]

        if not available_attributes:
            majority_class = max(set(y), key=y.count)
            return ["Leaf", majority_class, len(y), parent_sample_count]

        best_attr = self._choose_best_attribute(X, y, available_attributes)
        available_attributes = [attr for attr in available_attributes if attr != best_attr]
        tree = ["Attribute", "att"+str(best_attr)]
        unique_values = sorted(set(row[best_attr] for row in X))

        for value in unique_values:
            X_sub, y_sub = self._split_dataset(X, y, best_attr, value)
            if not y_sub:
                majority_class = max(set(y), key=y.count)
                tree.append(["Value", value, ["Leaf", majority_class, len(y_sub), len(y)]])
            else:
                subtree = self._tdidt(X_sub, y_sub, available_attributes.copy(), len(y))
                tree.append(["Value", value, subtree])

        return tree



    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            y_label = self._traverse_tree(self.tree, instance)
            if y_label is None:
                y_label = "A"
            y_predicted.append(y_label)
        return y_predicted

    def _traverse_tree(self, node, instance):
        if node[0] == "Leaf":
            return node[1]
        attr_index_name = node[1]
        attr_index = int(attr_index_name[3:])
        for branch in node[2:]:
            if branch[0] == "Value" and branch[1] == instance[attr_index]:
                return self._traverse_tree(branch[2], instance)
        return None

    def _choose_best_attribute(self, X, y, attributes):
        best_attr = None
        best_gain = -1
        base_entropy = self._entropy(y)

        for attr in attributes:
            sub_entropies = []
            unique_values = set(row[attr] for row in X)

            for value in unique_values:
                X_sub, y_sub = self._split_dataset(X, y, attr, value)
                if y_sub:
                    prob = len(y_sub) / len(y)
                    sub_entropies.append(prob * self._entropy(y_sub))

            info_gain = base_entropy - sum(sub_entropies)
            if info_gain > best_gain:
                best_gain = info_gain
                best_attr = attr

        return best_attr

    def _split_dataset(self, X, y, attr, value):
        X_sub = [row for row in X if row[attr] == value]
        y_sub = [y[i] for i in range(len(y)) if X[i][attr] == value]
        return X_sub, y_sub

    def _entropy(self, y):
        from math import log2
        entropy = 0
        total = len(y)
        for label in set(y):
            prob = y.count(label) / total
            entropy -= prob * log2(prob)
        return entropy

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rules = []
        self._traverse_tree_for_rules(self.tree, rules, [], attribute_names, class_name)
        print("\n".join(" AND ".join(rule[:-1]) + " Then "+ rule[-1] for rule in rules))
    def _traverse_tree_for_rules(self, node, rules, rule, attribute_names, class_name):
        if node[0] == "Leaf":
            rule.append(class_name + " = " + node[1])
            rules.append(rule)
            return
        attr_index_name = node[1]
        attr_index = int(attr_index_name[3:])
        for branch in node[2:]:
            new_rule = rule.copy()
            if attribute_names is None:
                new_rule.append("IF att" + str(attr_index) + " == " + str(branch[1]))
            else:
                new_rule.append("IF " + attribute_names[attr_index] + " == " + str(branch[1]))
            self._traverse_tree_for_rules(branch[2], rules, new_rule, attribute_names, class_name)



    # BONUS method
   
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """Visualizes a tree via the open source Graphviz package.

        Args:
            dot_fname (str): The name of the .dot output file.
            pdf_fname (str): The name of the .pdf output file generated from the .dot file.
            attribute_names (list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes).
        """
        try:
            from graphviz import Digraph
        except:
            pass
        dot = Digraph(comment='Decision Tree')

        def add_nodes_edges(node, parent_id=None):
            """Recursively add nodes and edges to the Digraph."""
            # Check if the current node is a leaf
            if node[0] == "Leaf":
                label = f"Predict: {node[1]}\nSamples: {node[2]}/{node[3]}"
                node_id = str(id(node))
                dot.node(node_id, label=label, shape='box')
                return node_id
            else:
                # Non-leaf node, show attribute
                attr_index = int(node[1][3:])  # Extract attribute index
                attr_name = attribute_names[attr_index] if attribute_names else f"att{attr_index}"
                label = f"{attr_name}?"
                node_id = str(id(node))
                dot.node(node_id, label=label)
        
            # Add edge from parent to current node
            # if parent_id is not None:
            #     dot.edge(parent_id, node_id)

            # Recursively add child nodes
            for branch in node[2:]:
                    branch_value = str(branch[1])
                    child_id = add_nodes_edges(branch[2], node_id)
                    dot.edge(node_id, child_id, label=branch_value)
            return node_id

        # Start adding nodes from the root
        add_nodes_edges(self.tree)

        # Save .dot file and generate .pdf file
        dot.render(dot_fname, format='pdf', outfile=pdf_fname)

class MyRandomForestClassifier:
    """Represents a Random Forest classifier.
    
    Attributes:
        n_trees(int): number of decision trees in the forest
        n_features_split(str or int): number of features to consider for each split
        trees(list): list of decision trees
    """
    
    def __init__(self, n_trees=10, n_features_split='sqrt', random_state=None):
        """Initialize the random forest with the given parameters.
        
        Args:
            n_trees(int): number of trees in the forest
            n_features_split(str or int): number of features to consider for each split
                ('sqrt' for square root of total features, 'log2' for log base 2,
                or int for specific number)
            random_state(int): random seed for reproducibility
        """
        self.n_trees = n_trees
        self.n_features_split = n_features_split
        self.random_state = random_state
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample from the training data."""
        n_samples = len(X)
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return [X[i] for i in idxs], [y[i] for i in idxs]
        
    def _get_n_features(self, n_total_features):
        """Determine number of features to consider at each split."""
        if isinstance(self.n_features_split, int):
            return min(self.n_features_split, n_total_features)
        elif self.n_features_split == 'sqrt':
            return max(1, int(np.sqrt(n_total_features)))
        elif self.n_features_split == 'log2':
            return max(1, int(np.log2(n_total_features)))
        else:
            return n_total_features
            
    def fit(self, X, y):
        """Fit the random forest to the training data.
        
        Args:
            X(list of list): The training instances
            y(list): The training labels
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_features = len(X[0]) if X else 0
        n_features_split = self._get_n_features(n_features)
        
        self.trees = []
        for _ in range(self.n_trees):
            # Create bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Train a decision tree on the bootstrap sample
            tree = MyDecisionTreeClassifier()
            tree.fit(X_sample, y_sample, n_features_split)
            self.trees.append(tree)
            
    def predict(self, X):
        """Make predictions using the random forest.
        
        Args:
            X(list of list): The instances to make predictions for
            
        Returns:
            list: The predicted labels
        """
        # Get predictions from each tree
        tree_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            tree_predictions.append(predictions)
            
        # Take majority vote for final predictions
        final_predictions = []
        for i in range(len(X)):
            instance_predictions = [pred[i] for pred in tree_predictions]
            # Get most common prediction
            final_predictions.append(Counter(instance_predictions).most_common(1)[0][0])
            
        return final_predictions

