import csv
import math


class DataSet:
    """
    This class reads the dataset from a csv file, given the file path as a string.
    It exposes the following class members:

        attributes: a list of strings representing the name of each attribute
        domains: a list of lists indicating the possible values each attribute
                 in self.attributes can take in the provided data
        examples: a list of lists, with each element representing a datapoint
    """
    def __init__(self, path_to_csv):
        with open(path_to_csv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            self.attributes = next(csvreader)
            self.examples = [row for row in csvreader]
            self.domains = [list(set(x)) for x in zip(*self.examples)]

    def set_attrs(self, attrs):
        self.attributes = attrs

    def set_examples(self, exs):
        self.examples = exs

    def set_domains(self, doms):
        self.domains = doms


class Node:
    """
    This class represents an internal node of a decision tree.
    `test_attr` is the index of the attribute to test at this node.
    `test_name` is the human-readable name of that attribute.
    The Node stores a dictionary `self.children` that maps values of the test
    attribute to subtrees, where each subtree is either a Node or a Leaf.
    """
    def __init__(self, test_attr, test_name=None):
        self.test_attr = test_attr
        self.test_name = test_name or test_attr
        self.children = {}

    def classify(self, example):
        """Classify an example based on its test attribute value."""
        return self.children[example[self.test_attr]].classify(example)

    def add_child(self, val, subtree):
        """Add a child node, which could be either a Node or a Leaf."""
        self.children[val] = subtree

    def show(self, level=1):
        """Print a human-readable representation of the tree"""
        print('Test:', self.test_name)
        for (val, subtree) in self.children.items():
            print(' ' * 4 * level, "if", self.test_name, '=', val, '==>', end=' ')
            if isinstance(subtree, Leaf):
                subtree.show()
            else:
                subtree.show(level + 1)

class Leaf:
    """A Leaf holds only a predicted class, with no test."""
    def __init__(self, pred_class):
        self.pred_class = pred_class

    def classify(self, example):
        return self.pred_class

    def show(self):
        """This will be called by the Node `show` function"""
        print('Predicted class:', self.pred_class)


def learn_decision_tree(dataset, target_name, feature_names, depth_limit):
    """
    Trains a decision tree on the provided dataset.
    The `target_name` parameter is the name of the attribute to be predicted.
    The `feature_names` are the names of input attributes that should be used to split the data.
    Finally, `depth_limit` is a parameter to control overfitting by cutting off the tree after
    a certain depth and predicting the plurality class at that split.

    This function should return a decision tree learned from the data.
    """
    domains = dataset.domains
    target = dataset.attributes.index(target_name) #index of the target attribute
    features = [dataset.attributes.index(name) for name in feature_names] #indices of attributes being used

    def decision_tree_learning(examples, attrs, parent_examples=(), depth=0):
        """
        This function signature is written to match the pseudocode
        on p. 702 of Russell and Norvig. We recommend following that
        pseudocode to implement your decision tree.
        Note that we are adding an argument for the current depth, so you can
        keep track of the depth limit.

        This function should return the decision tree that has been learned.
        """
        #death limit
        if depth == 0:
            return plurality_value(examples)
        
        if len(examples) == 0:
            return plurality_value(parent_examples)
        elif same_classification(examples) is not None:
            return same_classification(examples)
        elif len(attrs) == 0:
            return plurality_value(examples)
        else:
            A = argmax(attrs, examples)
            tree = Node(A, f"vote{A}")
            for v in [0, 1]: # 0 == Yea and 1 == Nay
                exs = split(A, examples)[v]
                temp_attrs = attrs[:]
                temp_attrs.remove(A)
                subtree = decision_tree_learning(exs, temp_attrs, examples, depth-1)
                if v == 0:
                    tree.add_child("Yea", subtree)
                else:
                    tree.add_child("Nay", subtree)
            return tree

    def split(attribute, examples):
        yes_examples = []
        no_examples = []
        for exs in examples:
            if exs[attribute] == 'Yea' or exs[attribute] == 'Present':
                yes_examples.append(exs)
            else:
                no_examples.append(exs)
        return [yes_examples, no_examples]

    def argmax(attributes, examples):
        entropy_attr = []
        for attribute in attributes:
            entropy_attr.append(information_gain(examples, split(attribute, examples)))

        return attributes[entropy_attr.index(max(entropy_attr))]

    def same_classification(examples):
        republicans = 0
        democrats = 0
        for exs in examples:
            if exs[target].lower() == "republican":
                republicans+=1
            else:
                democrats+=1   
        if republicans == 0:
            return Leaf("Democrats")
        elif democrats == 0:
            return Leaf("Republicans")
        else:
            return None

    def plurality_value(examples):
        republicans = 0
        democrats = 0
        for exs in examples:
            if exs[target].lower() == "republican":
                republicans+=1
            else:
                democrats+=1   
        return Leaf("Democrats") if democrats >= republicans else Leaf("Republicans")

    def entropy(examples):
        """Takes a list of examples and returns their entropy w.r.t. the target attribute"""
        republicans = 0
        democrats = 0
        for exs in examples:
            if exs[target].lower() == "republican":
                republicans+=1
            else:
                democrats+=1
        total = republicans + democrats
        p_republicans = republicans / total
        p_democrats = democrats / total
        
        log_dems = 0
        if p_democrats != 0:
            log_dems = math.log(p_democrats, 2)
        log_reps = 0
        if p_republicans != 0:
            log_reps = math.log(p_republicans, 2)

        return -p_democrats*log_dems - p_republicans*log_reps

    def information_gain(parent, children):
        """
        Takes a `parent` set and a subset `children` of the parent.
        Returns the information gain due to splitting `children` from `parent`.
        """
        e_children = 0
        for child in children:
            if len(child) == 0:
                continue
            e_children += len(child)/len(parent) * entropy(child)

        return entropy(parent) - e_children

    return decision_tree_learning(dataset.examples, features, depth=depth_limit)

if __name__ == '__main__':
    """
    You can use this area to test your implementation and to generate
    output for the assignment. The autograder will ignore this area.
    """

    ############################
    ###### Example usage: ######
    ############################

    data = DataSet("./congress_small.csv")

    # An example of learning a decision tree to predict party affiliation
    # based on the values of votes 4-7

    t = learn_decision_tree(
        data,
        "class",
        ["vote4", "vote5", "vote6", "vote7"],
        5
    )
    t.show()