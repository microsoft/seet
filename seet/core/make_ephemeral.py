"""ephemeral.py

Derived Node object that enables self-deletion once the object is out of
context. An object of type Node is often referred to by its parent Node, and
therefore is not necessarily released when it exits context. An EphemeralNode
object will be release when it exits context. Use with care!
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import types


def make_ephemeral(node_obj):
    """make_ephemeral.

    Transforms a node into an ephemeral object with lifetime managed by a
    scope. We create ephemeral objects to add temporary objects to the pose
    graph and avoid bloating.

    Example:
    root = Node()
    with make_ephemeral(Node(parent=root)) as ephemeral_child:
        ephemeral_grandchild = Node(parent=ephemeral_child)
        # Do work

    # ephemeral_child is no longer a child of root, which no longer holds a
    # reference to it; in due course, garbage collection will do its job.

    Args:
        node_obj (node.Node): instance of a Node object whose lifetime we seek
        to manage.

    Returns:
        _EphemeralNode: Derived Node type that works with context managers,
        i.e., can have scope-defined lifetimes controlled by a "with"
        statement.
    """

    class _EphemeralNode(node_obj.__class__):
        """_EphemeralNode.

        Derived Node object that self-deletes, together with all of its
        ephemeral children, once the object is out of context. We need this
        class because context-managed objects must have __enter__ and __exit__
        attributes as part of their type, not only in specific instances.
        """

        def __enter__(self):
            """__enter__.

            Setup of EphemeralNode does not do anything. However, if we write
            an __enter__ method for regular Node objects, than we should call
            super().__enter__(), as we would for any overwritten method.

            Returns:
                EphemeralNode: self.
            """

            return self

        def __exit__(self, type, value, traceback):
            """__exit__.

            Teardown of EphemeralNode removes its ephemeral children and
            removes itself from its parent node, if any.

            Args:
                type (ExceptionClass): class of exception, if passed.

                value (ExceptionType): type of exception, if passed.

                traceback (Traceback): traceback of exception, if passed.

            Returns:
                bool: Always returns false.
            """

            while len(self.children) > 0:
                child = self.children[0]
                make_ephemeral(child)
                child.__exit__(type, value, traceback)

            # What holds on a node as a resource is the reference from the node
            # to its parent. Once that is gone, regular garbage collection
            # takes over.
            if self.parent is not None:
                self.parent.remove_child(self)

            # If the Node class had an implementation of __exit__, we should
            # call it here instead.
            return False

    node_obj.__class__ = _EphemeralNode  # We need this.
    node_obj.__enter__ = types.MethodType(_EphemeralNode.__enter__, node_obj)
    node_obj.__exit__ = types.MethodType(_EphemeralNode.__exit__, node_obj)

    return node_obj
