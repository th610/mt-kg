
def draw_networkx_nodes(G, pos,
                        nodelist=None,
                        node_size=300,
                        node_color='r',
                        node_shape='o',
                        alpha=1.0,
                        cmap=None,
                        vmin=None,
                        vmax=None,
                        ax=None,
                        linewidths=None,
                        label=None,
                        **kwds):
    """Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    nodelist : list, optional
       Draw only specified nodes (default G.nodes())

    node_size : scalar or array
       Size of nodes (default=300).  If an array is specified it must be the
       same length as nodelist.

    node_color : color string, or array of floats
       Node color. Can be a single color format string (default='r'),
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    node_shape :  string
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8' (default='o').

    alpha : float
       The node transparency (default=1.0)

    cmap : Matplotlib colormap
       Colormap for mapping intensities of nodes (default=None)

    vmin,vmax : floats
       Minimum and maximum for node colormap scaling (default=None)

    linewidths : [None | scalar | sequence]
       Line width of symbol border (default =1.0)

    label : [None| string]
       Label for legend

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> nodes=nx.draw_networkx_nodes(G,pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    http://networkx.github.io/documentation/latest/gallery.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pyplot as plt
        import numpy
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if nodelist is None:
        nodelist = G.nodes()

    if not nodelist or len(nodelist) == 0:  # empty nodelist, no drawing
        return None

    try:
        xy = numpy.asarray([pos[v] for v in nodelist])
    except KeyError as e:
        raise ValueError('Node %s has no position.'%e)
    except ValueError:
        raise ValueError('Bad value in node positions.')

    node_collection = ax.scatter(xy[:, 0], xy[:, 1],
                                 s=node_size,
                                 c=node_color,
                                 marker=node_shape,
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax,
                                 alpha=alpha,
                                 linewidths=linewidths,
                                 label=label)

    node_collection.set_zorder(2)
    return node_collection

def draw_networkx_edges(G, pos,
                       edgelist=None,
                       width=1.0,
                       edge_color='k',
                       style='solid',
                       alpha=1.0,
                       edge_cmap=None,
                       edge_vmin=None,
                       edge_vmax=None,
                       ax=None,
                       arrows=True,
                       label=None,
                       **kwds):
    """Draw the edges of the graph G.
    This draws only the edges of the graph G.
    Parameters
    ----------
    G : graph
        A networkx graph
    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.
    edgelist : collection of edge tuples
        Draw only specified edges(default=G.edges())
    width : float, or array of floats
        Line width of edges (default=1.0)
    edge_color : color string, or array of floats
        Edge color. Can be a single color format string (default='k'),
        or a sequence of colors with the same length as edgelist.
        If numeric values are specified they will be mapped to
        colors using the edge_cmap and edge_vmin,edge_vmax parameters.
    style : string
        Edge line style (default='solid') (solid|dashed|dotted,dashdot)
    alpha : float
        The edge transparency (default=1.0)
    edge_cmap : Matplotlib colormap
        Colormap for mapping intensities of edges (default=None)
    edge_vmin,edge_vmax : floats
        Minimum and maximum for edge colormap scaling (default=None)
    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.
    arrows : bool, optional (default=True)
        For directed graphs, if True draw arrowheads.
    label : [None| string]
        Label for legend
    arrowsize : float, optional (default=20)
        Size of the arrow head (mutation_scale parameter)
    arrowstyle : string, optional (default='->')
        Arrow style (e.g., '->', '-|>', '<->', etc.)
    connectionstyle : string, optional (default='arc3')
        Connection style for arrows (e.g., 'arc3', 'angle3', etc.)
    Returns
    -------
    matplotlib.collection.LineCollection
        `LineCollection` of the edges
    Notes
    -----
    For directed graphs, proper arrows are drawn using matplotlib's 
    FancyArrowPatch. Arrow appearance can be customized using arrowsize,
    arrowstyle, and connectionstyle parameters. Arrows can be turned off 
    with keyword arrows=False.
    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> edges=nx.draw_networkx_edges(G,pos=nx.spring_layout(G))
    Also see the NetworkX drawing examples at
    http://networkx.github.io/documentation/latest/gallery.html
    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba, Colormap
        from matplotlib.collections import LineCollection
        import numpy as np
        from collections.abc import Iterable
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges())

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    if not isinstance(width, Iterable) or isinstance(width, str):
        lw = (width,)
    else:
        lw = width

    # Handle edge colors
    if (not isinstance(edge_color, str) and 
        isinstance(edge_color, Iterable) and 
        len(edge_color) == len(edge_pos)):
        
        if all(isinstance(c, str) for c in edge_color):
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([to_rgba(c, alpha) for c in edge_color])
        elif all(not isinstance(c, str) for c in edge_color):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if all(isinstance(c, Iterable) and len(c) in (3, 4) for c in edge_color):
                edge_colors = tuple(edge_color)
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            raise ValueError('edge_color must consist of either color names or numbers')
    else:
        if isinstance(edge_color, str) or len(edge_color) == 1:
            edge_colors = (to_rgba(edge_color, alpha),)
        else:
            raise ValueError('edge_color must be a single color or list of exactly m colors where m is the number of edges')

    edge_collection = LineCollection(edge_pos,
                                   colors=edge_colors,
                                   linewidths=lw,
                                   antialiaseds=(1,),
                                   linestyle=style,
                                   )
    edge_collection.set_zorder(1)  # edges go behind nodes
    edge_collection.set_label(label)
    ax.add_collection(edge_collection)

    # Note: there was a bug in mpl regarding the handling of alpha values for
    # each line in a LineCollection. It was fixed in matplotlib in r7184 and
    # r7189 (June 6 2009). We should then not set the alpha value globally,
    # since the user can instead provide per-edge alphas now. Only set it
    # globally if provided as a scalar.
    if np.isscalar(alpha):
        edge_collection.set_alpha(alpha)

    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, Colormap))
            edge_collection.set_array(np.asarray(edge_color))
            edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()

    arrow_collection = None
    if G.is_directed() and arrows:
        # Draw proper arrows using FancyArrowPatch
        from matplotlib.patches import FancyArrowPatch
        
        # Get arrow properties
        arrow_size = kwds.get('arrowsize', 20)
        arrow_style = kwds.get('arrowstyle', '->')
        connection_style = kwds.get('connectionstyle', 'arc3')
        
        # Create arrows for each edge
        for i, (src, dst) in enumerate(edge_pos):
            x1, y1 = src
            x2, y2 = dst
            
            # Skip if source and destination are the same
            if x1 == x2 and y1 == y2:
                continue
            
            # Get color for this edge
            if edge_colors is None:
                # Use single color or numeric mapping
                if isinstance(edge_color, str):
                    arrow_color = edge_color
                else:
                    arrow_color = 'k'  # default
            else:
                if i < len(edge_colors):
                    arrow_color = edge_colors[i]
                else:
                    arrow_color = edge_colors[0]
            
            # Get line width for this edge
            if isinstance(lw, (list, tuple, np.ndarray)):
                line_width = lw[i] if i < len(lw) else lw[0]
            else:
                line_width = lw
            
            # Create arrow patch
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle=arrow_style,
                                  shrinkA=5, shrinkB=5,
                                  mutation_scale=arrow_size,
                                  color=arrow_color,
                                  linewidth=line_width,
                                  linestyle=style,
                                  alpha=alpha,
                                  connectionstyle=connection_style)
            
            ax.add_patch(arrow)
        
        # Remove the LineCollection since we're using arrow patches
        # ax.collections.remove(edge_collection)
        # edge_collection = None

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    return edge_collection
def draw_networkx_labels(G, pos,
                        labels=None,
                        font_size=12,
                        font_color='k',
                        font_family='sans-serif',
                        font_weight='normal',
                        alpha=1.0,
                        bbox=None,
                        ax=None,
                        **kwds):
    """Draw node labels on the graph G.
    Parameters
    ----------
    G : graph
        A networkx graph
    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.
    labels : dictionary, optional (default=None)
        Node labels in a dictionary keyed by node of text labels
    font_size : int
        Font size for text labels (default=12)
    font_color : string
        Font color string (default='k' black)
    font_family : string
        Font family (default='sans-serif')
    font_weight : string
        Font weight (default='normal')
    alpha : float
        The text transparency (default=1.0)
    bbox : dict, optional
        Bounding box for text. If None, defaults to a light gray rounded 
        rectangle for better readability (default=None)
    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.
    **kwds : keyword arguments
        Additional keyword arguments for text rendering
    Returns
    -------
    dict
        `dict` of labels keyed on the nodes
    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> labels=nx.draw_networkx_labels(G,pos=nx.spring_layout(G))
    Also see the NetworkX drawing examples at
    http://networkx.github.io/documentation/latest/gallery.html
    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = dict((n, n) for n in G.nodes())

    # set optional alignment
    horizontalalignment = kwds.get('horizontalalignment', 'center')
    verticalalignment = kwds.get('verticalalignment', 'center')
    
    text_items = {}  # there is no text collection so we'll fake one
    
    # Default bbox style for better readability (like in your reference image)
    if bbox is None:
        bbox = dict(boxstyle="round,pad=0.2", 
                   facecolor='lightgray', 
                #    edgecolor='black',
                   alpha=0.1)
    
    for n, label in labels.items():
        (x, y) = pos[n]
        if not isinstance(label, str):
            label = str(label)  # this will cause "1" and 1 to be labeled the same
        
        t = ax.text(x, y,
                   label,
                   size=font_size,
                   color=font_color,
                   family=font_family,
                   weight=font_weight,
                   alpha=alpha,
                   horizontalalignment=horizontalalignment,
                   verticalalignment=verticalalignment,
                   transform=ax.transData,
                   bbox=bbox,
                   clip_on=True,
                   **kwds)
        text_items[n] = t
    
    return text_items