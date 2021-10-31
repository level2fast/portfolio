Imports System.IO
Public Class Form1
    Dim Nodelist As Array
    Dim mynodelist As Dictionary(Of Integer, Node) = New Dictionary(Of Integer, Node)
    Structure Node
        Dim id As Integer
        Dim neighbors As List(Of Node)
        Dim visited As Boolean
        'Dim distance As Integer
        Dim edge As Integer
    End Structure
    Dim visitedDictionary As Dictionary(Of Integer, Boolean) = New Dictionary(Of Integer, Boolean)
    Dim listOfNodes As List(Of Node) = New List(Of Node)

    Dim NodeTree As List(Of Node)
    Public Sub LoadNodes()
        Dim tempNode As Node = New Node()
        Dim addNeighbor As Node = New Node()
        Dim path As String = "..\..\Test.txt"
        Dim fileReader As String
        Dim value As Integer = 0
        Dim prevNodeId As Integer = -1

        Try
            For Each line As String In File.ReadLines(path)
                'Console.WriteLine(line)
                fileReader = line
                Nodelist = fileReader.Split(",")

                For value = 0 To Nodelist.Length
                    prevNodeId = Nodelist(0)
                    If (value = 0) Then
                        tempNode = New Node()
                        tempNode.neighbors = New List(Of Node)
                        tempNode.id = Nodelist(0)
                        ''tempNode.distance = 0
                        tempNode.visited = False
                        tempNode.edge = 0
                        If Not (visitedDictionary.ContainsKey(tempNode.id)) Then
                            visitedDictionary.Add(tempNode.id, tempNode.visited)
                        End If

                        If (mynodelist.ContainsKey(tempNode.id)) Then
                            'already in list
                            Continue For
                        Else
                            mynodelist.Add(tempNode.id, tempNode)
                            listOfNodes.Add(tempNode)
                        End If

                    ElseIf (value Mod 2 <> 0 And value <> Nodelist.Length) Then
                        tempNode = New Node()
                        tempNode.neighbors = New List(Of Node)
                        tempNode.visited = False
                        tempNode.id = Nodelist(value)
                        tempNode.edge = Nodelist(value + 1)
                        '  tempNode.distance = 0

                        If Not (visitedDictionary.ContainsKey(tempNode.id)) Then
                            visitedDictionary.Add(tempNode.id, tempNode.visited)
                        End If

                        'add neighbor nodes
                        addNeighbor = mynodelist(prevNodeId)
                        addNeighbor.neighbors.Add(tempNode)

                        'Console.WriteLine("addNeighbor.id " & addNeighbor.id)
                        'Console.WriteLine("tempNode.id " & tempNode.id)
                        'Console.WriteLine("tempNode.edge " & tempNode.edge)
                        'Console.WriteLine("tempNode.previousNode " & tempNode.previousNodeId)

                        If (mynodelist.ContainsKey(tempNode.id)) Then
                            'already in list
                            Continue For
                        Else
                            mynodelist.Add(tempNode.id, tempNode)
                            listOfNodes.Add(tempNode)
                        End If
                    End If

                Next
            Next
        Catch ex As Exception
            MsgBox(ex.ToString())
        End Try


        findShortestPath(mynodelist(1), mynodelist)


    End Sub
    Public Function findShortestPath(ByVal startNode As Node, ByVal mynodelist As Dictionary(Of Integer, Node)) As String
        ' Dim tempNeighbor As Neighbor
        Dim visited As List(Of Node) = New List(Of Node)
        Dim currentNode As Node = New Node()
        Dim tempDist As Integer = 0
        Dim priorityQ As Queue(Of Node) = New Queue(Of Node)
        Dim DistanceDict As Dictionary(Of Integer, Integer) = New Dictionary(Of Integer, Integer)

        startNode = mynodelist(startNode.id)

        priorityQ.Enqueue(startNode)

        For Each node In listOfNodes
            DistanceDict.Add(node.id, Integer.MaxValue)
        Next

        While priorityQ.Count > 0

            'deque node with the shortest distance
            currentNode = priorityQ.Dequeue()

            'if node has been visited skip this one
            If (visitedDictionary(currentNode.id)) Then
                Continue While
            End If

            'For the current node, consider all of its unvisited neighbors 
            For Each tempNeighbor In currentNode.neighbors

                If (DistanceDict(currentNode.id) = Integer.MaxValue) Then
                    'just assign the edge as the new distance since this is the first
                    'time the distance is being set for this node
                    tempDist = tempNeighbor.edge
                Else
                    'calculate their tentative distances through the current node
                    tempDist = tempNeighbor.edge + DistanceDict(currentNode.id)
                End If

                'Compare the newly calculated tentative distance to the current assigned value and assign the smaller one
                If (tempDist < DistanceDict(tempNeighbor.id)) Then
                    DistanceDict(tempNeighbor.id) = tempDist
                End If

                'insert neighbor into priority queue using distance for the priority
                priorityQ.Enqueue(mynodelist(tempNeighbor.id))
            Next

            'mark the current node as visited
            visitedDictionary(currentNode.id) = True

            'add this node to the visited list
            visited.Add(currentNode)

        End While
        Try
            'print shortest distance from source to all other nodes
            For Each tempNode In visited
                Console.WriteLine("Shoretest path from id: 1 to id: " & tempNode.id & " distance: " & DistanceDict.Item(tempNode.id))
                'For Each tempNeighbor In NodeTree.Item(tempNode.id).neighbors
                '    'if neighbor node is endNode and total path is shortest path we want to break loop

                'Next

            Next
            'print path from source to destination

        Catch ex As Exception
            MsgBox(ex.ToString())
        End Try


        Return ""
    End Function

    Private Sub ListBox1_SelectedIndexChanged(sender As Object, e As EventArgs) Handles ListBox1.SelectedIndexChanged

    End Sub

    Private Sub Form1_Load(sender As Object, e As EventArgs) Handles MyBase.Load
        LoadNodes()

    End Sub

    Private Sub Button1_Click(sender As Object, e As EventArgs) Handles Button1.Click

    End Sub
End Class
Public Class PriorityQueue(Of T)
    ' The items and priorities.
    Public Values As New List(Of T)()
    Public Priorities As New List(Of Integer)()

    ' Return the number of items in the queue.
    Public ReadOnly Property count() As Integer
        Get
            Return Values.Count
        End Get
    End Property

    ' Add an item to the queue.
    Public Sub Enqueue(ByVal new_value As T, ByVal _
        new_priority As Integer)
        Values.Add(new_value)
        Priorities.Add(new_priority)
    End Sub

    ' Remove the item with the smallest priority from the
    ' queue.
    Public Sub Find(ByRef top_value As T, ByRef _
        top_priority As Integer)
        ' Find the lowest priority.
        Dim best_index As Integer = 0
        top_priority = Priorities(0)
        For i As Integer = 1 To Priorities.Count - 1


            If (top_priority > Priorities(i)) Then
                top_priority = Priorities(i)
                best_index = i
            End If
        Next i

        ' Return the corresponding item.
        top_value = Values(best_index)


    End Sub

    ' Remove the item with the smallest priority from the
    ' queue.
    Public Sub Dequeue(ByRef top_value As T, ByRef _
        top_priority As Integer)
        ' Find the lowest priority.
        Dim best_index As Integer = 0
        top_priority = Priorities(0)
        For i As Integer = 1 To Priorities.Count - 1
            If (top_priority > Priorities(i)) Then
                top_priority = Priorities(i)
                best_index = i
            End If
        Next i

        ' Return the corresponding item.
        top_value = Values(best_index)

        ' Remove the item from the lists.
        Values.RemoveAt(best_index)
        Priorities.RemoveAt(best_index)
    End Sub
End Class
