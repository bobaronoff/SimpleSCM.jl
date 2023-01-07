module SimpleSCM
# module to create routines for directed acyclic graphs as structural causal models (i.e. scm)
# scm diagram is made of nodes(i.e. events) and edges. Edges are directred from cause to effect

using Random,Distributions
using Statistics, StatsModels, GLM
using Graphs, GraphRecipes, Plots
using DataFrames

#### export area ###

# export structures
export DistroParams, Distro, SimpleScmEvent

# begin export

# building functions
export SimpleScm, definedistribution
export add_scmevent!, delete_scmevent!, modify_scmevent!
export add_scmedge!, delete_scmedge!, modify_wts_scmedge!

# scm describing functions
export describe, names, events
# export eventtolabel, labeltoevent, scmindexoflabel, scmindexofevent

# semi-internal event lists
#export exogenouseventsscm, terminaleventsscm, scmcauses, scmeffects
#export commoncause, commoneffect, directeffect

# path functions
export pathsdirect, dependencypaths, dseparated, pathsbylabel
#export pathscommoncause, pathscommoneffect

# causal functions
export onlycausalopen, conditioningsets, independencesets

# plot functions
export plotscm

# sub-tree functions
#export allcausesubscm, alleffectsubscm

# simulation dataset functions
#export causerankscm 
export simulationdata, modelsimdf

# end export

# the bobdagnode structure is the basis of diagram
"""
    Distroparams is an internal structure to store distribution parameters.
"""
mutable struct DistroParams
    mean::Float64
    sd::Float64
end

"""
    Distro is structure to hold distribution type and parameters.
    Current distributions include 'normal' and 'binary'.
    These are used to generate simulation data.
    normal=> gaussian
    binary=> normal converted to 1/0 by percentile
"""
mutable struct Distro
    type::String
    params::DistroParams
end

"""
    SimpleScmEvent is the main structure for an event node.  A complete structural 
    model is defined by Vector{SimpleScmEvent}.
    Fields:
        event::Int64 - internal event node ID
        label::String - label for the event node. These are used for DAG and as
                        variable names for simulation data.  If first character is 
                        u or U then model assumes event is not measured.
        distribution::Distro - designation of target distribution for simulated data
        r_squared::Float64 - a number [0.1,0.9]; default is 0.9. Factors in unmeasured randomness in 
                            simulated data. r_squared is correlation between target distribution and
                            actual simulated data.
        causes::Vector{Int64} - a vector of event node ID with an edge that points into this event node.
        effects::Vecotr{Int64} - a vector of target event node ID with edge emminating from this event node.
        effectwts::Vector{Vector{Float64}} - each edge is assigned a weight (default is 1.0). These are used
                                            to generate simulation data when multiple edges enter a single
                                            event node.  They define the relative value of variance contribution.
"""
mutable struct SimpleScmEvent
    event::Int64
    label::String 
    distribution::Distro
    r_squared::Float64 
    causes::Vector{Int64}
    effects::Vector{Int64}
    effectwts::Vector{Vector{Float64}}
end

# new distro types will need adjustment of defaultdistroparams,definedistribution,printevents
# as well as routines to generate simulated data.

# new elements to struct SimpleScmEvent requires adjustment of add_scmevent! and modify_scmevent!
# would also need to adjust any functions that use that field to process i.e. simulationdata()

function defaultdistroparams()
    ddp=DistroParams(0.0,1.0)
    return ddp
end

"""
    definedistribution()

    This is function returns a Distro value defined by calling parameters.
        type=> 'normal' or 'binary' ; see docstring for Distro.
        mean::Float64 - mean value of simulated target distribution
        sd::Float64 - any value greater than 0.0; directs standard deviation of
                        simulated target distribution
    Default values for type 'normal' are mean =0.0 and sd=1.0
    Default values for type 'binary' do not exist. Must specify a mean proportion
        in range [0.05,0.95]

    Examples
    ```
    mydistro= definedistribution("normal")
    mydistro2= definedistribution("normal", mean=15, sd=3.5)
    mydistro3= definedistribution("binary", mean=.25)
    ```
    
"""
function definedistribution(type::String ; mean::Union{Int64,Float64}=0.0, sd::Union{Int64,Float64}=1.0)
    distrosupported=["normal","binary","null"]
    # binary is supported as normal converted to {0,1} via percentile.
    type=lowercase(type)
    if issubset([type],distrosupported)      
        # process and create Distro
        if type=="normal"
            if sd>0
                params=defaultdistroparams()
                params.mean=convert(Float64,mean)
                params.sd=convert(Float64,sd)
                dd=Distro(type,params)
                return dd
            else
                msg="normal distribution requires a std. dev. greater than zero."
                error(msg)
            end
        end
        if type=="binary"
            if mean>=0.05 && mean<=0.95
                params=defaultdistroparams()
                params.mean=convert(Float64,mean)
                params.sd=1.0
                dd=Distro(type,params)
                return dd
            else
                msg="binary distribution requires a mean probability between 0.05 and 0.95."
                error(msg)
            end
        end
        if type=="null"
            params=defaultdistroparams()
            dd=Distro(type,params)
            return dd
        end
    else
        msg="Requested distribution: '"*type*"' is not supported."
        error(msg)
    end
    return nothing
end

"""
    SimpleScm()

    This is a constructor for a structural causal model.

    Example:
    '''
    myscm=SimpleScm()
    '''

"""
function SimpleScm()
    return Vector{SimpleScmEvent}(undef,0)
end

# summarizing constructs as vectors
# events=[scm[i].event for i in eachindex(scm)]
# causes=[scm[i].causes for i in eachindex(scm)]
# effects=[scm[i].effects for i in eachindex(scm)]
# labels=[scm[i].label for i in eachindex(scm)]

"""
    add_scmevent!()

    This function adds an event node to a causal model i.e. Vector{SimpleScmEvent}
        and returns the assigned event ID.

    Parameters:
        scm::Vector{SimpleScmEvent} - a previously instantiated causal model.
        label::String - label for event node. This label is used in graphs and
                        to name variables in simulation data. If first character is 
                        (either) u or U the event is taken to mean 'unmeasured'.
        distribution::Distro (named and optional, default 'normal') - simulation is monotonically 
                        transformed into distribution specified.
        r_squared::Float64 (named and optional) -  a number [0.1,0.9]; default is 0.9 
                        Factors in unmeasured randomness in simulated data. 
                        r_squared is correlation between target distribution and
                        actual simulated data.

    Example:
    ```
    add_scmevent!(myscm,"Event A",r_squared=.9)
    add_scmevent!(mysc,"Event B",r_squared=.8,distribution=definedistribution("normal", mean=4,sd=2))
    add_scmevent!(myscm4,"Event C",r_squared=.1,distribution=definedistribution("binary", mean=.25))
    ```

"""
function add_scmevent!(scm::Vector{SimpleScmEvent}, label::String; 
                    distribution::Distro=definedistribution("normal"),r_squared::Float64=0.9)
    maxevent=0
    for i in eachindex(scm)
        temp=scm[i].event
        if temp>maxevent
            maxevent=temp
        end
    end
    nextevent=maxevent+1
    if r_squared< 0.1
        r_squared=0.1
    end
    if r_squared>0.9
        r_squared=0.9
    end
    #push!(scm,SimpleScmEvent(nextevent,[],[], label))
    push!(scm,SimpleScmEvent(nextevent, label,distribution ,r_squared, [],[],[]))
    return nextevent
end

"""
    delete_scmevent!()

    This function removes the specified event node from specified causal model (scm).
    The node can be specified by either it's event ID or label.

    Example
    ```
    delete_scmevent!(myscm,"Event A")

    ```

"""
function delete_scmevent!(scm::Vector{SimpleScmEvent},  event::Int64)
    events=[scm[i].event for i in eachindex(scm)]
    idx=findfirst(isequal.(events,event))
    if isnothing(idx)
        idx=0
    else
        #delete references in causes and effects
        causes=[scm[i].causes for i in eachindex(scm)]
        for i in eachindex(causes)
            ridx=findall(isequal.(causes[i],event))
            if !isnothing(ridx)
                deleteat!(causes[i],ridx)
            end
        end
        effects=[scm[i].effects for i in eachindex(scm)]
        for i in eachindex(effects)
            ridx=findall(isequal.(effects[i],event))
            if !isnothing(ridx)
                deleteat!(effects[i],ridx)
            end
        end
        deleteat!(scm,idx)

    end   
    return idx
end

function delete_scmevent!(scm::Vector{SimpleScmEvent},  dlabel::String)
    labels=[scm[i].label for i in eachindex(scm)]
    didx=findfirst(isequal.(dlabel,labels))
    if isnothing(didx)
        msg="Specified label not found: add_scmedge."
        error(msg)
    end
    delete_scmevent!(scm,scm[didx].event)
end

"""
    This function modifies parameters to an existing event node.
    Modifiable parameters include label, distribution, and r_squared.
    The event can be indicated by event ID or label.

    Example
    ```
    modify_scmevent!(myscm,"X", newlabel="newX")
    modify_scmevent!(myscm,"newX", distribution=definedistribution("binary",mean=.6))
    modify_scmevent!(myscm,"newX", r_squared=.5)
    modify_scmevent!(myscm,"newX",newlabel="oldX",distribution=definedistribution("normal"),r_squared=0.9))
    ```
    
"""
function modify_scmevent!(scm::Vector{SimpleScmEvent},  event::Int64 ; newlabel::String="xzkg513", 
    distribution::Distro=definedistribution("null"),r_squared::Float64=-666.0)

    
    events=[scm[i].event for i in eachindex(scm)]
    idx=findfirst(isequal.(events,event))
    if isnothing(idx)
        msg="Event ID not found - modify_scmevent!"
        error(msg)
    else
        if newlabel!="xzkg513"
            scm[idx].label=newlabel
        end
        if r_squared!=-666
            if r_squared< 0.1
                r_squared=0.1
            end
            if r_squared>0.9
                r_squared=0.9
            end
            scm[idx].r_squared=r_squared
        end
        if distribution.type != "null"
            scm[idx].distribution=distribution
        end
    end   
    return idx
end

function modify_scmevent!(scm::Vector{SimpleScmEvent},  mlabel::String ; newlabel::String="xzkg513",
    distribution::Distro=definedistribution("null"),r_squared::Float64=-666.0)

    labels=[scm[i].label for i in eachindex(scm)]
    midx=findfirst(isequal.(mlabel,labels))
    if isnothing(midx)
        msg="Specified label not found: modify_scmedge."
        error(msg)
    end
    modify_scmevent!(scm,scm[midx].event,newlabel=newlabel, distribution=distribution, r_squared=r_squared)
end

"""
    add_scmedge!()

    This function adds a new edge between existing event nodes.  Events can be
        specified by event ID or label. The direction of the edge is from the cause event 
        to effect event.  The weight (i.e. wt) parameter is used when simulating data. 
        When multiple nodes combine, weights are used to proportion variance contributions
        to the receiving node (based on the standard/normalized causative distributions).
        The default is wt=1.0

    Example
    ```
    add_scmedge!(myscm, "X1","X2",wt=1.5)
    add_scmedge!(myscm,"X2","X3")
    ```
    
"""
function add_scmedge!(scm::Vector{SimpleScmEvent}, cause::Int64, effect::Int64; wt::Vector{}=[1.0])
               # wts::Vector{Union{Int64,Float64}}=[1.0])
    #check if at least two events exist
    if length(scm)>1
        # effect can not equal cause
        if cause != effect
            # check if cause and effect events exist
            events=[scm[i].event for i in eachindex(scm)]
            causeidx=findfirst(isequal.(events,cause))
            effectidx=findfirst(isequal.(events,effect))
            if isnothing(causeidx) || isnothing(effectidx)
                msg = "Edge cause and/or effect events not present."
                error(msg)
                return 0
            end
            # check if edge exists
            causeout=scm[causeidx].effects
                #effectin=scm[effectidx].causes
            causedupe=findfirst(isequal.(causeout,effect))
            if !isnothing(causedupe)
                msg="Edge already exists."
                error(msg)
                return 0
            end
            # check that edge does not violate acyclic rules
            edgevalid=edgevalidate(scm,cause,effect)
            if edgevalid != 1
                msg="Proposed edge violates acyclic criteria."
                error(msg)
                return 0
            end
            
            # add edge to scm
            push!(scm[causeidx].effects,effect)
            push!(scm[effectidx].causes,cause)
            push!(scm[causeidx].effectwts,wt)
            
            return 1
        else
            msg ="Edge cause can not equal effect."
            error(msg)
            return 0
        end
    else
        msg="Can not add edge to empty DAG."
        error(msg)
        return nothing
    end
end

function add_scmedge!(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String; wt::Vector{}=[1.0])
    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified label(s) not found: add_scmedge."
        error(msg)
    end
    add_scmedge!(scm,scm[cidx].event,scm[eidx].event,wt=wt)
end

"""
    delete_scmedge!()

    This function will remove an existing edge between a 'cause' event and 'effect' event.
    Event nodes can be referenced by event ID or label.

    Example
    ```
    delete_scmedge!(myscm,"X2","X3")
    ```
    `
"""
function delete_scmedge!(scm::Vector{SimpleScmEvent}, cause::Int64, effect::Int64)
    # not yet written to handle errors
    events=[scm[i].event for i in eachindex(scm)]
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end
    effects=scm[eventidx[cause]].effects
    effectwts=scm[eventidx[cause]].effectwts
    eidx=findfirst(isequal.(effects,effect))
    if !isnothing(eidx)
        deleteat!(effects,eidx)
        deleteat!(effectwts,eidx)
    end
    scm[eventidx[cause]].effects=effects
    scm[eventidx[cause]].effectwts=effectwts
    causes=scm[eventidx[effect]].causes
    cidx=findfirst(isequal.(causes,cause))
    if !isnothing(cidx)
        deleteat!(causes,cidx)
    end
    scm[eventidx[effect]].causes=causes
    return 1
end

function delete_scmedge!(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String)
    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified labels not found: delete_scmedge."
        error(msg)
    end
    delete_scmedge!(scm,scm[cidx].event,scm[eidx].event)
end

"""
    modify_wts_scmedge!()

    This function modifies the 'wt' parameter of an existing edge.
    'cause' and 'event' nodes can be specified by event ID or label.

    Example
    ```
    modify_wts_scmedge!(myscm,"X1","X2", wt=.75)
    ```

"""
function modify_wts_scmedge!(scm::Vector{SimpleScmEvent}, cause::Int64, effect::Int64; wt::Vector{}=[1.0])
    #error handling not yet written
    events=[scm[i].event for i in eachindex(scm)]
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end
    effects=scm[eventidx[cause]].effects
    effectwts=scm[eventidx[cause]].effectwts
    eidx=findfirst(isequal.(effects,effect))
    if !isnothing(eidx)
        effectwts[eidx]=wt
    end
    scm[eventidx[cause]].effectwts=effectwts
    return 1
end

function modify_wts_scmedge!(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String; wt::Vector{}=[1.0])
    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified label(s) not found: modify_wts_scmedge."
        error(msg)
    end
    modify_wts_scmedge!(scm,scm[cidx].event,scm[eidx].event,wt=wt)
end

function causerankscm(scm::Vector{SimpleScmEvent})
    #ranks events(i.e. nodes) in a order for causation propagation
    #returns vector of scm indexes (i.e. not event ID)
    causes=[scm[i].causes for i in eachindex(scm)]
    pcs=deepcopy(causes)
    pidx=collect(1:length(causes))
    scmrank=Vector{Int64}(undef,0)
    eventrank=Vector{Int64}(undef,0)
    while length(pcs)>0
        # move each event with zero causes into rank and erank
        nc=length.(pcs)
        rc=findall(isequal.([0],nc))
        nrc=length(rc)
        if !isnothing(rc)
            append!(scmrank,pidx[rc])
            # append to eventrank
            for j in 1:nrc
              push!(eventrank,scm[pidx[rc[j]]].event)
            end
            deleteat!(pcs,rc)
            deleteat!(pidx,rc)
        end
        # loop through eventrank and remove these elements from pcs elements
        lc=Vector{Int64}(undef,0)
        for i in eachindex(pcs)
            if allainb(pcs[i],eventrank)
                push!(lc,i)
            end
        end
        for i in eachindex(lc)
            push!(eventrank,scm[pidx[lc[i]]].event)
            push!(scmrank,pidx[lc[i]])
        end
        deleteat!(pcs,lc)
        deleteat!(pidx,lc)
    end

    return scmrank,eventrank
end

function allainb(a::Vector{Int64},b::Vector{Int64})
    return (prod(in.(a,Ref(b))))
end

"""
    simulationdata()

    This function creates a DataFrame with simulated data based on parameters of the
        designated structural causal model.  Parameter 'nsims' is the number of rows 
        simulated; default rows=1000 . 
        There is a default 'randomseed' that can be changed.

    Example
    ```
    mydf= simulationdata(myscm, nsims=5000, randomseed=5551212
    ```

"""
function simulationdata(scm::Vector{SimpleScmEvent}, nsims::Int64=1000; randomseed::Int64=1234)
    Random.seed!(randomseed)
    # function to create simulation data
    # for time being, assume all data is Float64
    scmrank,eventrank = causerankscm(scm)
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(eventrank)
        eventidx[eventrank[i]]=scmrank[i]
    end
    nvar=length(scmrank)
    simdata=Array{Float64}(undef,(nsims,nvar))
    labeldata=Array{String}(undef,nvar)
    for i in eachindex(scmrank)
        causes=scm[scmrank[i]].causes
        if length(causes)==0
            distro=scm[scmrank[i]].distribution
            if distro.type in ["normal","binary"]
                # all data normalized at creation
                ytemp=rand(Normal(0,1),nsims)
                simdata[:,scmrank[i]]= ytemp
                labeldata[scmrank[i]]=scm[scmrank[i]].label
            end
            # future area for additional distributions
        else
            distro=scm[scmrank[i]].distribution
            if distro.type in ["normal","binary"]
                nc=length(causes)
                xvar=Array{Float64}(undef,(nsims,nc))
                betas=Array{Float64}(undef,(nc,1))
                for j in eachindex(causes)
                    xvar[:,j]=simdata[:, eventidx[causes[j]]]
                    tw=scm[eventidx[causes[j]]].effectwts
                    betas[j,1]=tw[1][1]
                end
                r2=scm[scmrank[i]].r_squared
                yhat1=xvar*betas
                varhat=var(yhat1)
                betas2 = sqrt(r2/varhat) .* betas
                ytemp=xvar*betas2
                if r2<1
                    varepsi=(1-r2)*var(ytemp)
                    epsi=rand(Normal(0,sqrt(varepsi)),nsims)
                    ytemp=epsi + ytemp
                end
                simdata[:,scmrank[i]]=ytemp
                labeldata[scmrank[i]]=scm[scmrank[i]].label
            end 
        end
    end
    
    # recenter column to target distro parameters 
    for i in 1:nvar
         if scm[scmrank[i]].distribution.type=="normal"
            tmn=scm[scmrank[i]].distribution.params.mean
            tsd=scm[scmrank[i]].distribution.params.sd
            simdata[:,scmrank[i]]= tsd .* simdata[:,scmrank[i]] .+ tmn
         end
         if scm[scmrank[i]].distribution.type=="binary"
            tmn=scm[scmrank[i]].distribution.params.mean
            cutq=quantile(simdata[:,scmrank[i]],(1.0-tmn))
            simdata[:,scmrank[i]] = convert.(Int64, simdata[:,scmrank[i]] .> cutq )
         end
    end  
    
    simdf=DataFrame()
    simdf=DataFrame(simdata, :auto)
    # convert binary to Int64
    for i in 1:nvar
        if scm[scmrank[i]].distribution.type=="binary"
            simdf[!,scmrank[i]]= convert.(Int64,simdf[!,scmrank[i]])
        end
    end

    # fix errant names
    labeldata=strip.(labeldata)
    labeldata=replace.(labeldata," "=>"_")
    labeldata=replace.(labeldata,"-"=>"_")
    labeldata=replace.(labeldata,"/"=>"_")
    rename!(simdf,labeldata)
    return simdf
end

"""
    modelsimdf()

    This function will create a model, linear vs logisitic, using the supplied parameters.
        The treatment/intervention and outcome event nodes can be specified using either
        event ID or label (i.e. clabel,elabel).  A conditioning set (i.e. cset) is supplied as
        vector of labels or event ID).  

        The model is processed by GLM.jl and is the return value.
            coefficient table = coeftable(model)
            table.col[4] gives p-values
            table.rownms gives row names
            table.colnms gives col names

    Example
    ```
    mymodel=modelsimdf(mydf,"X","Y", cset=["X2","X3])
    resultdata=coeftable(mymodel)
    ```

"""
function modelsimdf(simdf::DataFrame, clabel::String,elabel::String; 
                        cset::Vector{String}=Vector{String}(undef,0))
    # general formula = term(elabel)~term(1)+term(clabel)+sum(term.(cset))
    # (linear)model = lm(formula,df)
    # parameters/properties of lm provided by functions in ...
    # StatsBase.jl (https://juliastats.org/StatsBase.jl/latest/statmodels/)
    # i.e. r2 => r-squared
    # coefficient table = coeftable(model)
    # i.e. table.cols[4] gives p values
    # i.e.  table.rownms gives row names
    # i.e.  table.colnms gives col names

    #
    # coeftable(lm_fit) |> c -> c.cols[c.pvalcol][c.rownms .== "x"]
    # This will give the p-value for variable “x”.

    labeldata=[clabel,elabel]
    labeldata=strip.(labeldata)
    labeldata=replace.(labeldata," "=>"_")
    labeldata=replace.(labeldata,"-"=>"_")
    labeldata=replace.(labeldata,"/"=>"_")
    clabel=labeldata[1]
    elabel=labeldata[2]

    if length(cset)>0
        cformula = term(elabel)~term(1)+term(clabel)+sum(term.(cset))
    else
        cformula = term(elabel)~term(1)+term(clabel)
    end
    println(cformula)
    # check if linear or logistic
    tset=unique(sort(simdf[!,elabel]))
    if tset==[0,1]
        #logistic model
        simmodel = glm(cformula,simdf,Bernoulli(),LogitLink())
        return(simmodel)
    else
        #linear model
        simmodel = lm(cformula,simdf)
        return(simmodel)
    end
end

function modelsimdf(simdf::DataFrame, scm::Vector{SimpleScmEvent},event1::Int64, event2::Int64; 
    cset::Vector{Int64}=Vector{Int64}(undef,0))
    if length(cset)>0
        eset=vcat(event1,event2,cset)
        labeldata=eventtolabel(scm, eset)
        labeldata=strip.(labeldata)
        labeldata=replace.(labeldata," "=>"_")
        labeldata=replace.(labeldata,"-"=>"_")
        labeldata=replace.(labeldata,"/"=>"_")
        clabel=labeldata[1]
        elabel=labeldata[2]
        cset2=labeldata[3:end]
        return (modelsimdf(simdf,clabel,elabel,cset=cset2))
    else
        eset=vcat(event1,event2)
        labeldata=eventtolabel(scm, eset)
        labeldata=strip.(labeldata)
        labeldata=replace.(labeldata," "=>"_")
        labeldata=replace.(labeldata,"-"=>"_")
        labeldata=replace.(labeldata,"/"=>"_")
        clabel=labeldata[1]
        elabel=labeldata[2]
        return(modelsimdf(simdf,clabel,elabel))
    end
end

"""
    describe()

    This function prints the specifics of a structural model.  Each event node is 
        listed along with event ID, label, distribution, r-squared, causes (i.e. 
        'incoming arrows'), and effects (i.e. 'outgoing arrows')

    Example
    ```
    describe(myscm)
    ```

"""
function describe(scm::Vector{SimpleScmEvent})
    scmrank,eventrank = causerankscm(scm)
    # order by causal rank
    scm2=scm[scmrank]
    
    events=[scm2[i].event for i in eachindex(scm2)]
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end
    println
    #future: place sorting routine here and readjust looping
    for event in scm2
        # routine to print each node
        println(event.label * " (id= " * string(event.event) * ")")
        strparams=""
        if event.distribution.type=="normal"
            strparams=" (mean: " * string(event.distribution.params.mean) * " ; sd: " * 
                string(event.distribution.params.sd) * " )"
        end
        if event.distribution.type=="binary"
            strparams=" (mean: " * string(event.distribution.params.mean) * " )"
        end
        println("   " * "distribution: " * event.distribution.type * strparams)
        println("   " * "fit: R^2="*string(event.r_squared))
        print("   causes:")
        causes=event.causes
        nc=length(causes)
        if nc>0
            println()
            for i in 1:nc
                cidx=eventidx[causes[i]]
                cs=scm2[cidx].effects
                cw=scm2[cidx].effectwts
                csidx=findfirst(isequal.(event.event,cs))
                println("    - " * scm2[cidx].label * " (wt: "*string(cw[csidx])*")")
            end
        else
            println(" *none*")
        end
        print("   effects:")
        effects=event.effects
        nc=length(effects)
        if nc>0
            println()
            for i in 1:nc
                println("    - " * scm2[eventidx[effects[i]]].label * " (wt: "*string(event.effectwts[i])*")")
            end
        else
            println(" *none*")
        end
    end
end

"""
    names()

    This function returns a Vector{String} and lists the labels for each event node.

    Example
    ```
    eventnames=names(myscm)
    ```

"""
function names(scm::Vector{SimpleScmEvent})
    labels=[scm[i].label for i in eachindex(scm)]
    return labels
end

"""
    events()

    This function returns a dictionary.  The keys represent event ID's and the values
        are the corresponding event labels.  This is useful for mapping.

    Example
    ```
    labeldict=events(myscm)
    ```
"""
function events(scm::Vector{SimpleScmEvent})
    events=[scm[i].event for i in eachindex(scm)]
    labels=[scm[i].label for i in eachindex(scm)]
    labeldict=Dict{Int64,String}()
    
    for i in eachindex(events)
        labeldict[events[i]]=labels[i]
    end

    return(labeldict)
end

function eventtolabel(scm::Vector{SimpleScmEvent}, eventlist::Vector{Int64}) 
    ne=length(eventlist)
    if ne>0
        events=[scm[i].event for i in eachindex(scm)]
        labels=[scm[i].label for i in eachindex(scm)]
        labellist=Vector{String}(undef,ne)
        for i in 1:ne
            il=findfirst(isequal.(eventlist[i],events))
            if !isnothing(il)
                labellist[i]=labels[il]
            else
                msg="A member of event-ID list not found - eventtolabel."
                error(msg)
            end
        end
    else
        labellist=Vector{String}(undef,0)
    end
    return labellist
end

function labeltoevent(scm::Vector{SimpleScmEvent}, labellist::Vector{String}) 
    nl=length(labellist)
    if nl>0
        events=[scm[i].event for i in eachindex(scm)]
        labels=[scm[i].label for i in eachindex(scm)]
        eventlist=Vector{Int64}(undef,nl)
        for i in 1:nl
            il=findfirst(isequal.(labellist[i],labels))
            if !isnothing(il)
                eventlist[i]=events[il]
            else
                msg="A member of label list not found - label to event."
                error(msg)
            end
        end
    else
        eventlist=Vector{Int64}(undef,0)
    end
    return eventlist
end

function scmindexoflabel(scm::Vector{SimpleScmEvent}, label::Any)
    labels=[scm[i].label for i in eachindex(scm)]
    idx=findall(isequal.(labels,label))
    vidx=Vector{Tuple{Int64, Int64}}(undef,0)
    for i in eachindex(idx)
        push!(vidx,(idx[i],scm[idx[i]].event))
    end
    return vidx
end

function scmindexofevent(scm::Vector{SimpleScmEvent}, event::Int64)
    events=[scm[i].event for i in eachindex(scm)]
    idx=findfirst(isequal.(events,event))
    if isnothing(idx)
        idx=0
    end   
    return idx
end

function edgevalidate(scm::Vector{SimpleScmEvent}, cause::Int64, effect::Int64)
    # check that proposed edge does not create a cycle
    validity=1
    # create Dict(i.e. map) correlating event id to index location
    events=[scm[i].event for i in eachindex(scm)]
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end
    # enumerate effects events of proposed effect
    # if effects events include cause then criteria violated
    effectsevents=Vector{Int64}(undef,0)
    gen=Vector{Int64}(undef,0)
    push!(gen,effect)
    while length(gen)>0
        # add current generation to 'events and assemble next generation
        nextgen=Vector{Int64}(undef,0)
        for i in eachindex(gen)
            push!(effectsevents,gen[i])
            append!(nextgen, scm[eventidx[gen[i]]].effects )
        end
        gen=nextgen       
    end
    badedge=findfirst(isequal.(effectsevents,cause))
    if !isnothing(badedge)
        validity=0
    end
    if validity==1
        # now enumerate causes events from cause
        causesevents=Vector{Int64}(undef,0)
        gen=Vector{Int64}(undef,0)
        push!(gen,cause)
        while length(gen)>0
            # add current generation to 'events and assemble next generation
            nextgen=Vector{Int64}(undef,0)
            for i in eachindex(gen)
                push!(causesevents,gen[i])
                append!(nextgen, scm[eventidx[gen[i]]].causes )
            end
            gen=nextgen       
        end
        badedge=findfirst(isequal.(causesevents,effect))
        if !isnothing(badedge)
            validity=0
        end
    end    
    return validity
end

function exogenouseventsscm(scm::Vector{SimpleScmEvent})
    topevents=Vector{Int64}(undef,0)
    if length(scm)>0
        causes=[scm[i].causes for i in eachindex(scm)]
        events=[scm[i].event for i in eachindex(scm)]
        ncauses=length.(causes)
        tops=findall(==(0),ncauses)
        if length(tops)>0
            topevents=events[tops]
        end
    end
    return topevents
end

function terminaleventsscm(scm::Vector{SimpleScmEvent})
    bottomevents=Vector{Int64}(undef,0)
    if length(scm)>0
        effects=[scm[i].effects for i in eachindex(scm)]
        events=[scm[i].event for i in eachindex(scm)]
        neffects=length.(effects)
        tops=findall(==(0),neffects)
        if length(tops)>0
            bottomevents=events[tops]
        end
    end
    return bottomevents
end

# support functions

function scmcauses(scm::Vector{SimpleScmEvent}, event::Int64 ; dosort=true)
    # create Dict(i.e. map) correlating event id to index location
    events=[scm[i].event for i in eachindex(scm)]
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end
    # enumerate effects events 
    causesevents=Vector{Int64}(undef,0)
    gen=Vector{Int64}(undef,0)
    push!(gen,event)
    while length(gen)>0
        # add current generation to 'events and assemble next generation
        nextgen=Vector{Int64}(undef,0)
        for i in eachindex(gen)
            push!(causesevents,gen[i])
            append!(nextgen, scm[eventidx[gen[i]]].causes )
        end
        gen=nextgen       
    end
    if dosort
        #remove calling event from list
        popfirst!(causesevents)
        unique!(sort!(causesevents))
    else
        unique!(causesevents)
    end
    return causesevents
end

function scmeffects(scm::Vector{SimpleScmEvent}, event::Int64 ; dosort=true)
    # create Dict(i.e. map) correlating event id to index location
    events=[scm[i].event for i in eachindex(scm)]
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end
    # enumerate effects events 
    effectsevents=Vector{Int64}(undef,0)
    gen=Vector{Int64}(undef,0)
    push!(gen,event)
    while length(gen)>0
        # add current generation to 'events and assemble next generation
        nextgen=Vector{Int64}(undef,0)
        for i in eachindex(gen)
            push!(effectsevents,gen[i])
            append!(nextgen, scm[eventidx[gen[i]]].effects )
        end
        gen=nextgen       
    end
    if dosort
        #remove calling event from list
        popfirst!(effectsevents)
        unique!(sort!(effectsevents))
    else
        unique!(effectsevents)
    end
    return effectsevents
end

# functions to compare events relationship

function commoncause(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64)
    ccause=Vector{Int64}(undef,0)
    cause1=scmcauses(scm,event1)
    cause2=scmcauses(scm,event2)
    # remove event1 and event2 from consideration

    if length(cause1)>0 && length(cause2)>0
        ccause=cause1[findall(in(cause2),cause1)]
    end
    return ccause
end

function commoneffect(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64)
    ceffect=Vector{Int64}(undef,0)
    effect1=scmeffects(scm,event1)
    effect2=scmeffects(scm,event2)
    # remove event1 and event2 from consideration

    if length(effect1)>0 && length(effect2)>0
        ceffect=effect1[findall(in(effect2),effect1)]
    end
    return ceffect
end

function directeffect(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64)
    ce=Vector{Int64}(undef,0)
    # check if event 1 causes event2 
    ce1=scmeffects(scm,event1,dosort=false)
    if length(ce1)>0
        ce2=findfirst(isequal.(ce1,event2))
        if !isnothing(ce2)
            #event1 causes event2
            ce=ce1[1:ce2]
        end
    end
    if length(ce)==0
        # check if event2 causes event1
        ce1=scmeffects(scm,event2,dosort=false)
        if length(ce1)>0
            ce2=findfirst(isequal.(ce1,event1))
            if !isnothing(ce2)
                #event2 causes event1
                ce=ce1[1:ce2]
            end
        end
    end
    # ce contains side events not in the 'path'
    # may be more than one path
    return ce
end

# functions to create path(s) from commoncause,commoneffect, and directeffect lists

"""
    pathsdirect()

    This function returns a Vector{Vector{Int64}} where each index contains
        a direct causal path between first event node and second event node.
        Event nodes can be input as event ID or label.
    The returned paths are listed as event ID.

    Example
    ```
    directpaths=pathsdirect(myscm,"X","Y")
    directpathsbylabel=pathsbylabel(myscm,directpaths)
    
    ```

"""
function pathsdirect(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64)
    goodpaths=Vector{Vector{Int64}}(undef,0)
    if event1==event2
        push!(goodpaths,[event1])
        return goodpaths
    end
    directcheck=directeffect(scm,event1,event2)
    if length(directcheck)==0
        return goodpaths
    end
    event1=directcheck[1]
    event2=directcheck[end]
    # create Dict(i.e. map) correlating event id to index location
    events=[scm[i].event for i in eachindex(scm)]
    effects=[scm[i].effects for i in eachindex(scm)]
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end
    work1paths=Vector{Vector{Int64}}(undef,0)
    push!(work1paths,[event1])
    ncause=length(effects[eventidx[event1]])
    while ncause>0
        tgen=0
        work2paths=Vector{Vector{Int64}}(undef,0)
        for i in eachindex(work1paths)
            tevent=work1paths[i][end]
            gen=effects[eventidx[tevent]]
            tgen += length(gen)
            for j in eachindex(gen)
                if gen[j]==event2
                    gpath=vcat(work1paths[i],gen[j])
                    push!(goodpaths,gpath)
                else
                    gpath=vcat(work1paths[i],gen[j])
                    push!(work2paths,gpath)
                end
            end
        end
        work1paths= deepcopy(work2paths)
        ncause=tgen
    end
    return goodpaths
end

function pathsdirect(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String)
    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified labels not found: pathsdirect."
        error(msg)
    end
    pathsdirect(scm,scm[cidx].event,scm[eidx].event)
end

function pathsbylabel(scm::Vector{SimpleScmEvent},pathlist::Vector{Vector{Int64}})
    labeldict=events(scm)
    pathlabels= collect(map(x->labeldict[x],path) for path in pathlist)
    return pathlabels
end

"""
    dependencypaths()

    This function returns a Vector{NamedTuple}.  Each element represents a path of
        potential statistical dependency as defined by the structural model, between the
        first event node and second event node, and indicat by event ID or label.
    The named tuple includes:
        pathnodes => a vector of event nodes along the path (as event ID)
        nodetypes => a vector with each element corresponding to the event in pathnodes.
                     Node types include: HNI,HNO (handle in, handle out)
                                         CH (chained)
                                         CC (common cause)
                                         CL (collider)
        causal => a boolean (true/false) as whether path is 'causal'
        blocked => a boolean (true/false) as whether path is 'blocked' in 
                   non-conditioned state.

    Example
    ```
    dpaths=dependencypaths(myscm,"X","Y")
    ```

"""
function dependencypaths(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64)
    # function to find dependency paths as well as ....
    # calculate if dependnecy path is open or blocked
    
    goodpaths=Vector{Vector{Int64}}(undef,0)
    events=[scm[i].event for i in eachindex(scm)]
    effects=[scm[i].effects for i in eachindex(scm)]
    connections= [vcat(scm[i].effects,scm[i].causes) for i in eachindex(scm)]

    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end

    work1paths=Vector{Vector{Int64}}(undef,0)
    push!(work1paths,[event1])
    nconnect=length(work1paths)

    while nconnect>0
        work2paths=Vector{Vector{Int64}}(undef,0)
        for i in eachindex(work1paths)
            tevent=work1paths[i][end]
            gen=connections[eventidx[tevent]]
            for j in eachindex(gen)
                if gen[j]==event2 && !(issubset(gen[j],work1paths[i]))
                    gpath=vcat(work1paths[i],gen[j])
                    push!(goodpaths,gpath)
                elseif !(issubset(gen[j],work1paths[i]))
                    gpath=vcat(work1paths[i],gen[j])
                    push!(work2paths,gpath)
                end
            end
        end
        work1paths= deepcopy(work2paths)
        nconnect=length(work1paths)
    end
    # goodpaths is vector of paths where dependency could propagate

    nodecat=Vector{Vector{String}}(undef,0)
    for tpath in goodpaths
        ntpath=length(tpath)
        nodetype=fill("",ntpath)
        nodetype[1]=issubset(tpath[2],effects[eventidx[tpath[1]]]) ? "HNO" : "HNI"
        nodetype[end]=issubset(tpath[end],effects[eventidx[tpath[end-1]]]) ? "HNI" : "HNO"
        if ntpath>2
            larrow = issubset(tpath[2],effects[eventidx[tpath[1]]]) ? 1 : 0
            for i=2:(ntpath-1)
                rarrow = issubset(tpath[i],effects[eventidx[tpath[i+1]]]) ? 1 : 0
                if larrow != rarrow
                    nodetype[i]="CH"
                elseif larrow==1
                    nodetype[i]="CL"
                else
                    nodetype[i]="CC"
                end
                larrow=1-rarrow
            end
        end
        push!(nodecat,nodetype)
    end
    # nodecat is a vector of string vectors that categorizes each nodecat
    # HNI=> path handle into handle, HNO=> path handle out of handle,PC=>parent/child, 
    # CL=>collider, CC=>common cause, CH=>chain
    # if a path contains a CL node the path is blocked.
    # if all paths are blocked then event1 and event2 are d-separated (else d-connected)

    # check if dependency path is causal(i.e. event1 causes event2)
    #check if dependency path is naturally blocked

    causdep=fill(false,length(goodpaths))
    blocked=fill(false,length(goodpaths))
    for i in eachindex(goodpaths)
        tpath=nodecat[i]
        if length(tpath)>2
            innerpath=unique(tpath[2:(end-1)])
        else
            innerpath=["CH"]  # forces assessment as causal
        end
        if tpath[1]=="HNO" && tpath[end]=="HNI" && innerpath==["CH"]
            causdep[i]=true
        end
        if issubset(["CL"],innerpath)
            blocked[i]=true
        end
    end
    
    dependpaths=
      NamedTuple{(:pathnodes, :nodetypes, :causal, :blocked)}.(tuple.(goodpaths, nodecat,causdep,blocked))
    return dependpaths
end

function dependencypaths(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String)
    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified labels not found: delete_scmedge."
        error(msg)
    end
    dependencypaths(scm,scm[cidx].event,scm[eidx].event)
end

"""
    dseparated()

    This function returns a boolean (true/false) as whether two event nodes are
        d-separated.  Event nodes can be repesented by event ID or label.

    Example
    ```
    separationstatus=dseparated(myscm,"X","Y")
    ```

"""
function dseparated(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64 ; 
                            cset::Union{Vector{Int},Vector{String}}=Vector{Int64}(undef,0))
    # determines d-separation between two nodes and condition set
    # for cset to be empty it needs to be typed a Vector{Int64}
    dsep=true
    cset=unique(cset)
    lcset=length(cset)
    if lcset>0
        if typeof(cset)==Vector{String}
            events=[scm[i].event for i in eachindex(scm)]
            labels=[scm[i].label for i in eachindex(scm)]
            cset2=Vector{Int64}(undef,lcset)
            for i in 1:lcset
                ic=findfirst(isequal.([cset[i]],labels))
                if !isnothing(ic)
                    cset2[i]=events[ic]
                else
                    msg="Condition set label: '" * cset[i] * "' not found - dseparated."
                    error(msg)
                end
            end
            cset=cset2
        end
        badc=findfirst(isequal.([event1],cset))
        if !isnothing(badc)
            deleteat!(cset,badc)
        end
        badc=findfirst(isequal.([event2],cset))
        if !isnothing(badc)
            deleteat!(cset,badc)
        end
    end
    dpaths=dependencypaths(scm,event1,event2)
    if length(dpaths)>0
        # pivot on return tuple
        pathnodes=[x[1] for x in dpaths ]
        nodetypes=[x[2] for x in dpaths ]
        blocked=[x[4] for x in dpaths ]
        if length(cset)>0
            #need to reprocess blocked status given the condition set
            blocked=fill(false,length(nodetypes))
            for i in eachindex(nodetypes)
                ttype=nodetypes[i]
                tevent=pathnodes[i]
                if length(ttype)>2
                    ttype2=ttype[2:(end-1)]
                    tevent2=tevent[2:(end-1)]
                    for j in eachindex(ttype2)
                        if issubset([tevent2[j]],cset)
                            if ttype2[j]=="CL"
                                ttype2[j]="CH"
                            else
                                ttype2[j]="CL"
                            end
                        else
                            #need to see if descendant of a collider is in cset
                            if ttype2[j]=="CL"
                                dnodes=scmeffects(scm,tevent2[j])
                                inboth=intersect(dnodes,cset)
                                if length(inboth)>0
                                    ttype2[j]="CH"
                                end
                            end
                        end                      
                    end
                    innerpath=unique(ttype2)
                else
                    innerpath=["CH"]  # forces assessment as causal
                end
                if issubset(["CL"],innerpath)
                    blocked[i]=true
                end
            end
            if sum(blocked)<length(blocked)
                dsep=false
            end
            return dsep
        else
            if sum(blocked)<length(blocked)
                dsep=false
            end
            return dsep
        end
    else
        return dsep
    end
end

function dseparated(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String ; 
                        cset::Union{Vector{Int},Vector{String}}=Vector{Int64}(undef,0))
    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified labels not found: dseparated."
        error(msg)
    end
    dseparated(scm, scm[cidx].event, scm[eidx].event,cset=cset)
end

"""
    onlycausalopen()

    This function returns a boolean (true/fale) whether all non-causal paths
        are blocked.  A conditioning set (of event nodes) can specified (i.e. cset).
        cset must be input in form of a Vector.  The elements can indicate either
        event ID or label.

    Example
    ```
    causalonly=onlycausalopen(myscm,"X","Y")
    causalonly2=onlycausalopen(myscm,"X","Y", cset=["Z1","Z2","Z3"])

    ```
    
"""
function onlycausalopen(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64 ; 
                            cset::Union{Vector{Int},Vector{String}}=Vector{Int64}(undef,0),
                            verbose::Bool=false, bylabel::Bool=true)
    # determines if paths between event1 and event2 have only unblocked causal and
    # blocked non-causal
    # for cset to be empty it needs to be typed a Vector{Int64}

    events=[scm[i].event for i in eachindex(scm)]
    labels=[scm[i].label for i in eachindex(scm)]
    labeldict=Dict{Int64,String}()
    for i in eachindex(events)
        labeldict[events[i]]=labels[i]
    end

    onlycausal=false
    cset=unique(cset)
    lcset=length(cset)
    if lcset>0
        if typeof(cset)==Vector{String}
            events=[scm[i].event for i in eachindex(scm)]
            labels=[scm[i].label for i in eachindex(scm)]
            cset2=Vector{Int64}(undef,lcset)
            for i in 1:lcset
                ic=findfirst(isequal.([cset[i]],labels))
                if !isnothing(ic)
                    cset2[i]=events[ic]
                else
                    msg="Condition set label: '" * cset[i] * "' not found - onlycausalopen."
                    error(msg)
                end
            end
            cset=cset2
        end
        badc=findfirst(isequal.([event1],cset))
        if !isnothing(badc)
            deleteat!(cset,badc)
        end
        badc=findfirst(isequal.([event2],cset))
        if !isnothing(badc)
            deleteat!(cset,badc)
        end
    end
    dpaths=dependencypaths(scm,event1,event2)
    if length(dpaths)>0
        # pivot on return tuple
        pathnodes=[x[1] for x in dpaths ]
        nodetypes=[x[2] for x in dpaths ]
        causal=[x[3] for x in dpaths ]
        blocked=[x[4] for x in dpaths ]
        if length(cset)>0
            #adjust paths for conditioned events
            #need to reprocess blocked status given the condition set
            blocked=fill(false,length(nodetypes))
            causal=fill(false,length(nodetypes))
            for i in eachindex(nodetypes)
                ttype=nodetypes[i]
                tevent=pathnodes[i]
                for j in eachindex(ttype)
                    if issubset([tevent[j]],cset)
                        if ttype[j]=="CL"
                            ttype[j]="CC"
                        else
                            ttype[j]="CL"
                        end
                    else
                        #need to see if descendant of a collider is in cset
                        if ttype[j]=="CL"
                            dnodes=scmeffects(scm,tevent[j])
                            inboth=intersect(dnodes,cset)
                            if length(inboth)>0
                                ttype[j]="CH"
                            end
                        end
                    end
                end
                if length(ttype)>2
                    innerpath=unique(ttype[2:(end-1)])
                else
                    innerpath=["CH"]  # forces assessment as causal
                end
                if ttype[1]=="HNO" && ttype[end]=="HNI" && innerpath==["CH"]
                    causal[i]=true
                end
                if issubset(["CL"],innerpath)
                    blocked[i]=true
                end
            end
            if sum(causal)>0 && sum(isequal.(causal,blocked))==0
                onlycausal=true
            else
                onlycausal=false
                return onlycausal 
            end
        else
            # cset is empty
            if sum(causal)>0 && sum(isequal.(causal,blocked))==0
                onlycausal=true
                return onlycausal
            else
                onlycausal=false
                if verbose
                    cb00=findall((.!causal) .& (.!blocked))
                    ncb00=length(cb00)
                    if ncb00>0
                        println("\nUnblocked non-causal paths include:\n")
                        problempaths=pathnodes[cb00]
                        if bylabel
                            problempaths2=map.(x->labeldict[x],problempaths)
                            for i in 1:ncb00
                                println(problempaths2[i])
                            end
                        else
                            for i in 1:ncb00
                                println(problempaths[i])
                            end
                        end
                        println()
                    end
                end
                return onlycausal 
            end
        end

    else
        return onlycausal
    end

end

function onlycausalopen(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String ; 
    cset::Union{Vector{Int},Vector{String}}=Vector{Int64}(undef,0),
    verbose::Bool=false, bylabel::Bool=true )

    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified labels not found: dseparated."
        error(msg)
    end
    onlycausalopen(scm, scm[cidx].event, scm[eidx].event,cset=cset, verbose=verbose, bylabel=bylabel)
end

function pathscommoncause(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64)
    goodpaths=Vector{Vector{Int64}}(undef,0)
    if event1==event2
        push!(goodpaths,[event1])
        return goodpaths
    end
    cccheck=commoncause(scm,event1,event2)
    ncommon=length(cccheck)
    if ncommon==0
        return goodpaths
    end
    for i in eachindex(cccheck)
        leg1paths=pathsdirect(scm,cccheck[i],event1)
        leg2paths=pathsdirect(scm,cccheck[i],event2)
        if length(leg1paths)>0 && length(leg2paths)>0
            for j in eachindex(leg1paths) , k in eachindex(leg2paths)
                leg3path=reverse(leg1paths[j][2:end])
                dupevents=intersect(leg3path,leg2paths[k])
                if length(dupevents)==0
                    newpath=vcat(leg3path,leg2paths[k])
                    push!(goodpaths,newpath)
                end
            end
        end
    end
    return goodpaths
end

function pathscommoneffect(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64)
    goodpaths=Vector{Vector{Int64}}(undef,0)
    if event1==event2
        push!(goodpaths,[event1])
        return goodpaths
    end
    cccheck=commoneffect(scm,event1,event2)
    ncommon=length(cccheck)
    if ncommon==0
        return goodpaths
    end 
    for i in eachindex(cccheck)
        leg1paths=pathsdirect(scm,event1,cccheck[i])
        leg2paths=pathsdirect(scm,event2,cccheck[i])
        if length(leg1paths)>0 && length(leg2paths)>0
            for j in eachindex(leg1paths) , k in eachindex(leg2paths)
                leg3path=reverse(leg2paths[k][1:(end-1)])
                dupevents=intersect(leg3path,leg1paths[j])
                if length(dupevents)==0
                    newpath=vcat(leg1paths[j],leg3path)
                    push!(goodpaths,newpath)
                end
            end
        end
    end
    return goodpaths

end

"""
    conditioningsets()

    This function returns a Vector{Vector{Union{Int64,String}}}.  Each element contains
        a conditioning set of event nodes that will block all non-causal paths between 
        first and second events.  These events can be specified by event ID or label.
        The condition sets will be in the same form (event ID vs label) as index events.
        This behavior can be over-ridden with the 'bylabel' parameter set to 'true'.
    The 'verbose' parameter indicates if progress is printed.

    Example
    ```
    csets=conditioningsets(myscm,"X","Y")
    ```

"""
function conditioningsets(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64; 
    verbose::Bool=true , bylabel::Bool=false)
    # functions to create conditioning sets
    conditionsets=Vector{Vector{String}}(undef,0)
    mtset=Vector{Int64}(undef,0)
    reliable=true
    dpaths=dependencypaths(scm,event1,event2)
    npaths=length(dpaths)
    if npaths>0
        # global definitions
        events=[scm[i].event for i in eachindex(scm)]
        labels=[scm[i].label for i in eachindex(scm)]
        labeldict=Dict{Int64,String}()
        for i in eachindex(events)
            labeldict[events[i]]=labels[i]
        end
        cl2desc=Dict{Int64,Vector{Int64}}()
        desc2cl=Dict{Int64,Vector{Int64}}()
        eventblockers=Dict{Int64,Vector{Vector{Int64}}}()

        # path processing
        pathnodes=[x[1] for x in dpaths ]
        nodetypes=[x[2] for x in dpaths ]
        causal=[x[3] for x in dpaths ]
        blocked=[x[4] for x in dpaths ]
        cb10=findall(causal .& (.!blocked))
        cb00=findall((.!causal) .& (.!blocked))
        cb01=findall((.!causal) .& (blocked))

        # look for 'early' return
        if length(cb10)<1
            if verbose
                println("\nNo causal paths exist !!!\n")
            end
        reliable=false
        return(conditionsets)
        end
        if length(cb00)==0
            if verbose
                println("\nNo unblocked non-causal paths exist !!!\n")
                println("System entered already causal !!!\n")
            end
            push!(conditionsets, mtset)
            return(conditionsets)
        end

        # classify events
        causalevents= unique(reduce(vcat,pathnodes[cb10]))
        # create list of unmeasured and therefore not conditional contenders
        unmeasured=events[findall(collect( lowercase(i[1])=='u' for i in labels))]
        unmeasured=setdiff(unmeasured,[event1,event2])
        modelpathevents=setdiff(unique(reduce(vcat,pathnodes)),[event1,event2])
        modelunmeasured=intersect(unmeasured,modelpathevents)
        modelmeasured=setdiff(modelpathevents, modelunmeasured)
        

        modeltypes=reduce(vcat,nodetypes)
        modelevents=reduce(vcat,pathnodes)
        modelchain=unique(modelevents[findall(isequal.(["CH"],modeltypes))])
        modelcomcause=unique(modelevents[findall(isequal.(["CC"],modeltypes))])
        modelchcc=unique(vcat(modelchain,modelcomcause))
        modelcolliders=unique(modelevents[findall(isequal.(["CL"],modeltypes))])
        modeltypes=unique(modeltypes)
        modelevents=unique(modelevents)
        modelneutral=setdiff(modelmeasured,modelcolliders)
        modelneutral=setdiff(modelneutral,causalevents)
        modelneutral=setdiff(modelneutral,[event1,event2])

        # catelog colliders/descendants
        for  i in eachindex(modelcolliders)
            cldesc=scmeffects(scm,modelcolliders[i])
            #cldesc=setdiff(cldesc,modelevents)
            cl2desc[modelcolliders[i]]= unique(vcat(get(cl2desc,modelcolliders[i],mtset),cldesc))
            for j in eachindex(cldesc)
                desc2cl[cldesc[j]]=unique(vcat(get(desc2cl,cldesc[j],mtset),modelcolliders[i]))
            end
        end
        
        # catalog dual role events
        modelschizo1=intersect(modelchcc,modelcolliders)
  
        # derive safe event blockers
        for i in eachindex(modelevents)
            if in(modelevents[i],modelunmeasured)
                eventblockers[modelevents[i]]=[mtset]
                continue
            end
            if in(modelevents[i],causalevents)
                eventblockers[modelevents[i]]=[mtset]
                continue
            end
            if in(modelevents[i],modelchcc) && !in(modelevents[i],modelschizo1)
                eventblockers[modelevents[i]]=[[modelevents[i]]]
                continue
            end
            if in(modelevents[i],modelcolliders) && !in(modelevents[i],modelschizo1)
                eventblockers[modelevents[i]]=[mtset]
                continue
            end
            if in(modelevents[i],modelschizo1)
               #complex scenario 
               #need combination sets that block all cb01 where modelevent[i] has role of CL
               cb01nodes=pathnodes[cb01]
               cb01types=nodetypes[cb01]
               cb01idx=Vector{Int64}(undef,0)
               for j in eachindex(cb01nodes)
                for k in eachindex(cb01nodes[j])
                    if cb01nodes[j][k]==modelevents[i] && cb01types[j][k]=="CL"
                        push!(cb01idx,j)
                        continue
                    end
                end
               end
               conditionoptions=Vector{Vector{Int64}}(undef,0)
               push!(conditionoptions,[modelevents[i]])
               # list of blocked paths affected if condition on modelevent[i]
               cb01nodes=cb01nodes[cb01idx]
               for j in eachindex(cb01nodes)
                toptions=intersect(cb01nodes[j],modelneutral)
                if length(toptions)==0
                    if verbose
                        tevent=labeldict[modelevents[i]]
                        println("Unable to resolve complex event with dual role: ",tevent)
                    end
                    return(conditionsets)
                end
                push!(conditionoptions,toptions)
               end
                tset=collect.(vec(collect(Iterators.product(conditionoptions[1]))))
                ncb00=length(conditionoptions)
                if ncb00>1
                    for i in 2:ncb00
                        tset=collect.(vec(collect(Iterators.product(tset,conditionoptions[i])))) 
                    end
                    for i in 2:ncb00
                        tset=reduce.(vcat,tset)
                    end
                end
                tset=sort.(tset)
                tset=unique.(tset)
                tset=unique(tset)
                #sort!(tset, by=length)
                sort!(tset, by= x-> length(x)-length(intersect(x,modelcomcause))/10) #place sets with CC first
                eventblockers[modelevents[i]]=tset
                continue
            end
        end
        # eventblocker complete
        # cycle through cb00 and assemble conditioning list(s)
        conditionoptions=Vector{Vector{Int64}}(undef,0)
        
        for i in eachindex(cb00)
            pthoptions=Vector{Vector{Int64}}(undef,0)
            tpath=pathnodes[cb00[i]]
            ltpath=length(tpath)
            for j in 1:ltpath
                teb=eventblockers[tpath[j]]
                pthoptions=vcat(pthoptions,teb)
            end
            pthoptions=unique(pthoptions)
            deleteat!(pthoptions,findall(isequal.(0,length.(pthoptions))))
            conditionoptions=vcat(conditionoptions,[pthoptions])
        end
        tset=collect.(vec(collect(Iterators.product(conditionoptions[1]))))
        ncb00=length(conditionoptions)
        if ncb00>1
            for i in 2:ncb00
                tset=collect.(vec(collect(Iterators.product(tset,conditionoptions[i])))) 
            end
        end 
        for i in 1:ncb00
            tset=reduce.(vcat,tset)
        end
        tset=sort.(tset)
        tset=unique.(tset)
        tset=unique(tset)
        #sort!(tset, by=length)
        sort!(tset, by= x-> length(x)-length(intersect(x,modelcomcause))/10) #place sets with CC first
        
        ##### return results in proper form
        if bylabel
            tset2=map.(x->labeldict[x],tset)
            conditionsets=tset2
        else
            conditionsets=tset
        end
        return(conditionsets)

    else
        if verbose
            println("\nEvents are not connected !!!\n")
        end
        return(conditionsets) 
    end

end

function conditioningsets(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String; 
                verbose::Bool=true , bylabel::Bool=true)
    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified labels not found: conditioningsets."
        error(msg)
    end
    conditioningsets(scm,scm[cidx].event,scm[eidx].event, verbose=verbose,bylabel=bylabel)
end

"""
    independencesets()

    This function returns a Vector{Vector{Union{Int64,String}}}.  Each element contains
        a conditioning set of event nodes that will block all paths between first and 
        second events.  These events can be specified by event ID or label.
        The condition sets will be in the same form (event ID vs label) as index events.
        This behavior can be over-ridden with the 'bylabel' parameter set to 'true'.
    The 'verbose' parameter indicates if progress is printed.  The 'rehab' parameter
        specifies if function should attempt to resolve issues causes by an event node with
        dual roles across paths (i.e. collider and chain/common cause). This is best left
        at default value of 'true'.
    
    Example
    ```
    csets=independencesets(myscm,"X","Y")
     ```
       
"""
function independencesets(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64; 
    verbose::Bool=true , bylabel::Bool=false)
    # functions to create conditioning sets
    conditionsets=Vector{Vector{String}}(undef,0)
    mtset=Vector{Int64}(undef,0)
    reliable=true
    dpaths=dependencypaths(scm,event1,event2)
    npaths=length(dpaths)
    if npaths>0
        # global definitions
        events=[scm[i].event for i in eachindex(scm)]
        labels=[scm[i].label for i in eachindex(scm)]
        labeldict=Dict{Int64,String}()
        for i in eachindex(events)
            labeldict[events[i]]=labels[i]
        end
        cl2desc=Dict{Int64,Vector{Int64}}()
        desc2cl=Dict{Int64,Vector{Int64}}()
        eventblockers=Dict{Int64,Vector{Vector{Int64}}}()

        # path processing
        pathnodes=[x[1] for x in dpaths ]
        nodetypes=[x[2] for x in dpaths ]
        blocked=[x[4] for x in dpaths ]
        cb00=findall((.!blocked))
        cb01=findall((blocked))

        # look for 'early' return
        if length(cb00)==0
            if verbose
                println("\nNo unblocked non-causal paths exist !!!\n")
            end
            push!(conditionsets, mtset)
            return(conditionsets)
        end

        # classify eventssets
        unmeasured=events[findall(collect( lowercase(i[1])=='u' for i in labels))]
        unmeasured=setdiff(unmeasured,[event1,event2])
        modelpathevents=setdiff(unique(reduce(vcat,pathnodes)),[event1,event2])
        modelunmeasured=intersect(unmeasured,modelpathevents)
        modelmeasured=setdiff(modelpathevents, modelunmeasured)
        

        modeltypes=reduce(vcat,nodetypes)
        modelevents=reduce(vcat,pathnodes)
        modelchain=unique(modelevents[findall(isequal.(["CH"],modeltypes))])
        modelcomcause=unique(modelevents[findall(isequal.(["CC"],modeltypes))])
        modelchcc=unique(vcat(modelchain,modelcomcause))
        modelcolliders=unique(modelevents[findall(isequal.(["CL"],modeltypes))])
        modeltypes=unique(modeltypes)
        modelevents=unique(modelevents)
        modelneutral=setdiff(modelmeasured,modelcolliders)
        modelneutral=setdiff(modelneutral,[event1,event2])

        # catelog colliders/descendants
        for  i in eachindex(modelcolliders)
            cldesc=scmeffects(scm,modelcolliders[i])
            #cldesc=setdiff(cldesc,modelevents)
            cl2desc[modelcolliders[i]]= unique(vcat(get(cl2desc,modelcolliders[i],mtset),cldesc))
            for j in eachindex(cldesc)
                desc2cl[cldesc[j]]=unique(vcat(get(desc2cl,cldesc[j],mtset),modelcolliders[i]))
            end
        end
        
        # catalog dual role events
        modelschizo1=intersect(modelchcc,modelcolliders)
        
        # derive safe event blockers
        for i in eachindex(modelevents)
            if in(modelevents[i],modelunmeasured)
                eventblockers[modelevents[i]]=[mtset]
                continue
            end
            if in(modelevents[i],modelchcc) && !in(modelevents[i],modelschizo1)
                eventblockers[modelevents[i]]=[[modelevents[i]]]
                continue
            end
            if in(modelevents[i],modelcolliders) && !in(modelevents[i],modelschizo1)
                eventblockers[modelevents[i]]=[mtset]
                continue
            end
            if in(modelevents[i],modelschizo1)
               #complex scenario 
               #need combination sets that block all cb01 where modelevent[i] has role of CL
               cb01nodes=pathnodes[cb01]
               cb01types=nodetypes[cb01]
               cb01idx=Vector{Int64}(undef,0)
               for j in eachindex(cb01nodes)
                for k in eachindex(cb01nodes[j])
                    if cb01nodes[j][k]==modelevents[i] && cb01types[j][k]=="CL"
                        push!(cb01idx,j)
                        continue
                    end
                end
               end
               conditionoptions=Vector{Vector{Int64}}(undef,0)
               push!(conditionoptions,[modelevents[i]])
               # list of blocked paths affected if condition on modelevent[i]
               cb01nodes=cb01nodes[cb01idx]
               for j in eachindex(cb01nodes)
                toptions=intersect(cb01nodes[j],modelneutral)
                if length(toptions)==0
                    if verbose
                        tevent=labeldict[modelevents[i]]
                        println("Unable to resolve complex event with dual role: ",tevent)
                    end
                    return(conditionsets)
                end
                push!(conditionoptions,toptions)
               end
                tset=collect.(vec(collect(Iterators.product(conditionoptions[1]))))
                ncb00=length(conditionoptions)
                if ncb00>1
                    for i in 2:ncb00
                        tset=collect.(vec(collect(Iterators.product(tset,conditionoptions[i])))) 
                    end
                    for i in 2:ncb00
                        tset=reduce.(vcat,tset)
                    end
                end
                tset=sort.(tset)
                tset=unique.(tset)
                tset=unique(tset)
                #sort!(tset, by=length)
                sort!(tset, by= x-> length(x)-length(intersect(x,modelcomcause))/10) #place sets with CC first
                eventblockers[modelevents[i]]=tset
                continue
            end
            eventblockers[modelevents[i]]=[mtset]
        end
        # eventblocker complete
        # cycle through cb00 and assemble conditioning list(s)
        conditionoptions=Vector{Vector{Int64}}(undef,0)
        
        for i in eachindex(cb00)
            pthoptions=Vector{Vector{Int64}}(undef,0)
            tpath=pathnodes[cb00[i]]
            ltpath=length(tpath)
            for j in 1:ltpath
                teb=eventblockers[tpath[j]]
                pthoptions=vcat(pthoptions,teb)
            end
            pthoptions=unique(pthoptions)
            deleteat!(pthoptions,findall(isequal.(0,length.(pthoptions))))
            conditionoptions=vcat(conditionoptions,[pthoptions])
        end
        tset=collect.(vec(collect(Iterators.product(conditionoptions[1]))))
        ncb00=length(conditionoptions)
        if ncb00>1
            for i in 2:ncb00
                tset=collect.(vec(collect(Iterators.product(tset,conditionoptions[i])))) 
            end
        end 
        for i in 1:ncb00
            tset=reduce.(vcat,tset)
        end
        tset=sort.(tset)
        tset=unique.(tset)
        tset=unique(tset)
        #sort!(tset, by=length)
        sort!(tset, by= x-> length(x)-length(intersect(x,modelcomcause))/10) #place sets with CC first
        
        ##### return results in proper form
        if bylabel
            tset2=map.(x->labeldict[x],tset)
            conditionsets=tset2
        else
            conditionsets=tset
        end
        return(conditionsets)

    else
        if verbose
            println("\nEvents are not connected and therefore independent!!!\n")
        end
        push!(conditionsets,mtset)
        return(conditionsets) 
    end

end

function independencesets(scm::Vector{SimpleScmEvent}, clabel::String, elabel::String; 
    verbose::Bool=true , bylabel::Bool=true)

    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified labels not found: conditioningsets."
        error(msg)
    end
    independencesets(scm,scm[cidx].event,scm[eidx].event, verbose=verbose,bylabel=bylabel)
end

"""
    plotscm()

    This function will plot a DAG (i.e. directed acyclic graph) specified by the
        structural model.  If supplied with 'intervention' and 'outcome' nodes, dependency
        paths will be color coded.  These event nodes can be specified by event ID or label.
        Nodes will also be color coded (green => intervention/outcome, red => conditioned,
        gray => unmeasured.) Paths are color coded (green => causal and open,
        red => non-causal and blocked, yellow => non-causal and unblocked,
        purple => causal and blocked, blue => descendant of collider, 
        black => non-dependency path).
    If intervention/outcome are specified, this function will also accept a set of 
        conditioning event nodes.  The color coding of the paths will reflect the result 
        of conditioning.  The condition set (i.e. 'cset') must be input as a vector.  
        The elements of cset can be event ID or label.
    The named parameter 'title' allows speicifation of title for the plot.
    The named parameter 'coderender' is a boolean which will trigger an incode rendering of the plot.

    Example
    ```
    myplot=plotscm(myscm, title="My marvelous model DAG")
    myplot2=plotscm(myscm,"X","Y",cset=["Z1","Z2"], coderender=true)
    ```

"""
function plotscm(scm::Vector{SimpleScmEvent}; coderender::Bool=false, title::String="")
    # routine to plot a directed acyclic graphs
    scmrank,eventrank = causerankscm(scm)

    events=[scm[i].event for i in eachindex(scm)]
    effects=[scm[i].effects for i in eachindex(scm)]
    labels=[scm[i].label for i in eachindex(scm)]
    events=events[scmrank]
    effects=effects[scmrank]
    labels=labels[scmrank]

    nevents=length(events)
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end

    scmgraf=DiGraph()
    add_vertices!(scmgraf,nevents)
    wtnodes=fill(1.5,nevents)
    for i in eachindex(events)
        currentv1=eventidx[events[i]]
        currentv2= map.(x->eventidx[x],effects[i])
        for j in eachindex(currentv2)
            add_edge!(scmgraf,currentv1,currentv2[j])
        end
    end

    scmplot=graphplot(scmgraf, names=labels,
                        plot_title=title, plot_titlefontsize=18, 
                        method=:stress, root=:left,
                        size=(1200,1000), lw=3, curves=false ,
                        fontsize=20, nodeshape=:rect,
                        nodecolor=:lightcyan1)
    if coderender
        display(scmplot)
    end

    return(scmplot)
end

function plotscm(scm::Vector{SimpleScmEvent}, event1::Int64, event2::Int64; 
                 cset::Union{Vector{Int64},Vector{String}}=Vector{Int64}(undef,0), 
                 coderender::Bool=false, title::String="")
    # plot SCM with color coded paths
    scmrank,eventrank = causerankscm(scm)

    events=[scm[i].event for i in eachindex(scm)]
    effects=[scm[i].effects for i in eachindex(scm)]
    labels=[scm[i].label for i in eachindex(scm)]
    
    # cleanup cset
    cset=unique(cset)
    lcset=length(cset)
    if lcset>0
        if typeof(cset)==Vector{String}
            events=[scm[i].event for i in eachindex(scm)]
            labels=[scm[i].label for i in eachindex(scm)]
            cset2=Vector{Int64}(undef,lcset)
            for i in 1:lcset
                ic=findfirst(isequal.([cset[i]],labels))
                if !isnothing(ic)
                    cset2[i]=events[ic]
                else
                    msg="Condition set label: '" * cset[i] * "' not found - onlycausalopen."
                    error(msg)
                end
            end
            cset=cset2
        end
        badc=findfirst(isequal.([event1],cset))
        if !isnothing(badc)
            deleteat!(cset,badc)
        end
        badc=findfirst(isequal.([event2],cset))
        if !isnothing(badc)
            deleteat!(cset,badc)
        end
    end
    # end cset clean-up
    events=events[scmrank]
    effects=effects[scmrank]
    labels=labels[scmrank]
    edgecolorsidk=Dict{Tuple{Int64,Int64},Int64}()
    anomalies=Vector{Tuple{Int64,Int64}}(undef,0)

    dpaths=dependencypaths(scm,event1,event2)
    pathnodes=[x[1] for x in dpaths ]
    nodetypes=[x[2] for x in dpaths ]
    causal=[x[3] for x in dpaths ]
    blocked=[x[4] for x in dpaths ]

    pathnodes2=deepcopy(pathnodes)
    nodetypes2=deepcopy(nodetypes)
    causal2=deepcopy(causal)

    ######### begin process path status
    if length(cset)>0
        #adjust paths for conditioned events
        #need to reprocess blocked status given the condition set
        blocked=fill(false,length(nodetypes))
        causal=fill(false,length(nodetypes))
        for i in eachindex(nodetypes)
            ttype=nodetypes[i]
            tevent=pathnodes[i]
            for j in eachindex(ttype)
                if issubset([tevent[j]],cset)
                    if ttype[j]=="CL"
                        ttype[j]="CC"
                    else
                        ttype[j]="CL"
                    end
                else
                    #need to see if descendant of a collider is in cset
                    if ttype[j]=="CL"
                        dnodes=scmeffects(scm,tevent[j])
                        inboth=intersect(dnodes,cset)
                        if length(inboth)>0
                            ttype[j]="CH"
                        end
                    end
                end
            end
            if length(ttype)>2
                innerpath=unique(ttype[2:(end-1)])
            else
                innerpath=["CH"]  # forces assessment as causal
            end
            if ttype[1]=="HNO" && ttype[end]=="HNI" && innerpath==["CH"]
                causal[i]=true
            end
            if issubset(["CL"],innerpath)
                blocked[i]=true
            end
        end
    end
    
    ######### end process path status
    nevents=length(events)
    eventidx=Dict{Int64,Int64}()
    for i in eachindex(events)
        eventidx[events[i]]=i
    end

    #### build edge status dictionary
    for i in eachindex(pathnodes)
        cb=2*causal2[i] + 4 - blocked[i]
        npth=length(pathnodes[i])-1
        for j in 1:npth
            d1= eventidx[min(pathnodes[i][j] , pathnodes[i][j+1])]
            d2= eventidx[max(pathnodes[i][j] , pathnodes[i][j+1])]
            edgecolorsidk[(d1,d2)]= max(cb,get(edgecolorsidk,(d1,d2),1))
            edgecolorsidk[(d2,d1)]= edgecolorsidk[(d1,d2)]
        end
    end

    ## deal with collider descendants
    modeltypes=reduce(vcat,nodetypes2)
    modelevents=reduce(vcat,pathnodes2)
    modelcolliders=unique(modelevents[findall(isequal.(["CL"],modeltypes))])
    colpaths=Vector{Vector{Int64}}(undef,0)
    for i in eachindex(modelcolliders)
        cdnodes=scmeffects(scm,modelcolliders[i])
        for k in eachindex(cdnodes)
            append!(colpaths,pathsdirect(scm,modelcolliders[i],cdnodes[k]))
        end
    end
    unique!(colpaths)
    for i in eachindex(colpaths)
        pboth=colpaths[i]
        npb=length(pboth)-1
        for k in 1:npb
            push!(anomalies,(pboth[k],pboth[k+1]))
        end
    end
    for i in eachindex(anomalies)
        d1=eventidx[min(anomalies[i][1],anomalies[i][2])]
        d2=eventidx[max(anomalies[i][1],anomalies[i][2])]
        edgecolorsidk[(d1,d2)]= max(2,get(edgecolorsidk,(d1,d2),1))
        edgecolorsidk[(d2,d1)]= edgecolorsidk[(d1,d2)]
    end
    
    #### end edge status dictionary

    scmgraf=DiGraph()
    add_vertices!(scmgraf,nevents)
    for i in eachindex(events)
        currentv1=eventidx[events[i]]
        currentv2= map.(x->eventidx[x],effects[i])
        for j in eachindex(currentv2)
            add_edge!(scmgraf,currentv1,currentv2[j])
            clr1=min(currentv1,currentv2[j])
            clr2=max(currentv1,currentv2[j])
            edgecolorsidk[(clr1,clr2)]= max(1,get(edgecolorsidk,(clr1,clr2),1))
            edgecolorsidk[(clr2,clr1)]=edgecolorsidk[(clr1,clr2)]
        end
    end

    edgecolorpal=[:gray0,:steelblue1,:firebrick2,:darkgoldenrod1,:fuchsia,:green2 ]
    edgecolorsdict=Dict{Tuple{Int64,Int64},Symbol}()
    edgewidthpal=[3,3,3,6,6,3 ]
    edgewidthdict=Dict{Tuple{Int64,Int64},Int64}()
    for (key,value) in edgecolorsidk
        k1=key[1]
        k2=key[2]
        sval=edgecolorpal[value]
        edgecolorsdict[(k1,k2)]=sval
        lval=edgewidthpal[value]
        edgewidthdict[(k1,k2)]=lval
    end

    nodecolors=fill(:lightcyan1,length(labels))
    if length(cset)>0
        cnodes=map.(x->eventidx[x],cset)
        nodecolors[cnodes].=:pink
    end
    nunmeasured=findall(collect( lowercase(i[1])=='u' for i in labels))
    if length(nunmeasured)>0
        nodecolors[nunmeasured].=:ivory2
    end
    nodecolors[eventidx[event1]]=:seagreen1
    nodecolors[eventidx[event2]]=:seagreen1


    
    scmplot=graphplot(scmgraf, names=labels, 
                        plot_title=title, plot_titlefontsize=24, 
                        method=:spring, root=:left,
                        size=(1200,1000), edgewidth=edgewidthdict, curves=false ,
                        fontsize=20, nodeshape=:rect,
                        nodecolor=nodecolors,
                        edgecolor= edgecolorsdict )
    if sum(causal2)>0
        if (4 in values(edgecolorsidk))
            plot!(scmplot,annotation=((0.0,0.025),"UNBLOCKED NON-CAUSAL PATHS REQUIRE ATTENTION."), 
                                        annotationcolor=edgecolorpal[4],annotationfontsize=20,
                                        annotationhalign=:left , subplot=1)
        end
        if (6 in values(edgecolorsidk)) && !in(4,values(edgecolorsidk)) && !in(5,values(edgecolorsidk))
            plot!(scmplot,annotation=((0.0,0.025),"SUCCESSFUL ISOLATION OF CAUSAL EFFECTS."), 
                                        annotationcolor=edgecolorpal[6],annotationfontsize=24,
                                        annotationhalign=:left , subplot=1)
        end
        if (5 in values(edgecolorsidk))
            plot!(scmplot,annotation=((0.0,0.06),"BLOCKED CAUSAL PATHS REQUIRE ATTENTION."), 
                                        annotationcolor=edgecolorpal[5],annotationfontsize=20,
                                        annotationhalign=:left , subplot=1)
        end
    else
        plot!(scmplot,annotation=((0.0,0.06),"THERE ARE NO DIRECT CAUSAL PATHS TO ISOLATE."), 
                                        annotationcolor=edgecolorpal[3],annotationfontsize=24,
                                        annotationhalign=:left , subplot=1)
        if (4 in values(edgecolorsidk))
            plot!(scmplot,annotation=((0.0,0.025),"UNBLOCKED NON-CAUSAL PATHS ARE PRESENT."), 
                                        annotationcolor=edgecolorpal[4],annotationfontsize=20,
                                        annotationhalign=:left , subplot=1)
        end
    end
    if coderender
        display(scmplot)
    end

    return(scmplot )
end

function plotscm(scm::Vector{SimpleScmEvent},  clabel::String,  elabel::String ; 
                    cset::Union{Vector{Int64},Vector{String}}=Vector{Int64}(undef,0),
                    coderender::Bool=false, title::String="")

    labels=[scm[i].label for i in eachindex(scm)]
    cidx=findfirst(isequal.(clabel,labels))
    eidx=findfirst(isequal.(elabel,labels))
    if isnothing(cidx) || isnothing(eidx)
        msg="Specified labels not found: conditioningsets."
        error(msg)
    end
    
    plotscm(scm,scm[cidx].event,scm[eidx].event, cset=cset,coderender=coderender, title=title) 
end


# functions for subdiagrams

function allcausesubscm(scm::Vector{SimpleScmEvent}, event::Int64)
    events=[scm[i].event for i in eachindex(scm)]
    idx=findfirst(isequal.(events,event))
    if isnothing(idx)
        msg="All cause sug-DAG failed. Reference event not found."
        error(msg)
    end
    subscm=deepcopy(scm)
    keepevents=scmcauses(scm,event, dosort=false)
    removeevents=setdiff(events,keepevents)
    for i in eachindex(removeevents)
        delete_scmevent!(subscm,removeevents[i])
    end
    return subscm
end

function alleffectsubscm(scm::Vector{SimpleScmEvent}, event::Int64)
    events=[scm[i].event for i in eachindex(scm)]
    idx=findfirst(isequal.(events,event))
    if isnothing(idx)
        msg="All effect sug-DAG failed. Reference event not found."
        error(msg)
    end
    subscm=deepcopy(scm)
    keepevents=scmeffects(scm,event, dosort=false)
    removeevents=setdiff(events,keepevents)
    for i in eachindex(removeevents)
        delete_scmevent!(subscm,removeevents[i])
    end
    return subscm
end

# function allainb(a::Vector{Int64},b::Vector{Int64})
#     return (prod(in.(a,Ref(b))))
# end


end  # end module SimpleDFutils


