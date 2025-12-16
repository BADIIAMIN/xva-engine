===========================
Architecture & Design View
===========================

High-Level Component Diagram
============================

.. uml::

   @startuml XVA_Engine_HighLevel

   title XVA / CCR / PFE Engine - High Level Architecture

   skinparam componentStyle rectangle

   package "Configuration" {
     [RunConfig]
   }

   package "Orchestration" {
     [EngineRunner]
     [WorkflowManager]
   }

   package "Market Data" {
     [MarketDataEnvironment]
     [MarketDataSource]
   }

   package "Models & Simulation" {
     [RiskFactorModel]
     [CorrelationModel]
     [SimulationDriver]
     [TimeGrid]
     [RiskFactorCube]
   }

   package "Instruments & Pricing" {
     [Instrument]
     [Portfolio]
     [PricingContext]
     [PricingEngine]
     [PortfolioPricer]
   }

   package "Collateral & Aggregation" {
     [CSA]
     [CollateralEngine]
     [ExposureCube]
     [ExposureMetrics]
     [XVAEngine]
   }

   package "Validation & Backtesting" {
     [Backtest]
     [StatsTests]
     [PricerBenchmark]
   }

   package "Reporting" {
     [ReportExporter]
     [DocGenerator]
   }

   [EngineRunner] --> [RunConfig]
   [EngineRunner] --> [MarketDataEnvironment]
   [EngineRunner] --> [SimulationDriver]
   [EngineRunner] --> [PortfolioPricer]
   [EngineRunner] --> [CollateralEngine]
   [EngineRunner] --> [ExposureMetrics]
   [EngineRunner] --> [XVAEngine]
   [EngineRunner] --> [Backtest]
   [EngineRunner] --> [ReportExporter]

   [MarketDataEnvironment] <-- [MarketDataSource]

   [SimulationDriver] --> [RiskFactorModel]
   [SimulationDriver] --> [CorrelationModel]
   [SimulationDriver] --> [TimeGrid]
   [SimulationDriver] --> [RiskFactorCube]

   [PortfolioPricer] --> [Instrument]
   [PortfolioPricer] --> [Portfolio]
   [PortfolioPricer] --> [PricingEngine]
   [PortfolioPricer] --> [PricingContext]
   [PortfolioPricer] --> [RiskFactorCube]
   [PortfolioPricer] --> [ExposureCube]

   [CollateralEngine] --> [CSA]
   [CollateralEngine] --> [ExposureCube]

   [ExposureMetrics] --> [ExposureCube]
   [XVAEngine] --> [ExposureCube]

   [Backtest] --> [RiskFactorCube]
   [Backtest] --> [ExposureCube]
   [Backtest] --> [StatsTests]
   [PricerBenchmark] --> [PricingEngine]

   [ReportExporter] --> [ExposureCube]
   [ReportExporter] --> [RiskFactorCube]

   @enduml


Core Class Diagram
==================

.. uml::

   @startuml Core_ClassDiagram

   title Core API - Main Classes & Relations

   class RunConfig
   class EngineRunner

   class MarketDataEnvironment
   interface MarketDataSource

   class TimeGrid
   class RiskFactorCube
   class ExposureCube

   interface RiskFactorModel
   class CorrelationModel
   class SimulationDriver

   abstract class Instrument
   class Portfolio
   class PricingContext
   interface PricingEngine
   class PortfolioPricer

   class CSA
   class CollateralEngine
   class ExposureMetrics
   class XVAEngine

   interface Backtest
   class StatsTests
   class ReportExporter

   EngineRunner --> RunConfig
   EngineRunner --> MarketDataSource
   EngineRunner --> MarketDataEnvironment
   EngineRunner --> SimulationDriver
   EngineRunner --> PortfolioPricer
   EngineRunner --> CollateralEngine
   EngineRunner --> ExposureMetrics
   EngineRunner --> XVAEngine
   EngineRunner --> Backtest
   EngineRunner --> ReportExporter

   @enduml


Orchestration Sequence
======================

.. uml::

   @startuml EngineRunner_Sequence

   title XVA / PFE Run - Orchestration Flow

   actor User
   participant EngineRunner
   participant ConfigLoader
   participant MarketDataSource
   participant SimulationDriver
   participant PortfolioPricer
   participant CollateralEngine
   participant ExposureMetrics
   participant XVAEngine
   participant Backtest
   participant ReportExporter

   User -> EngineRunner: run("run_config.yaml")

   EngineRunner -> ConfigLoader: load("run_config.yaml")
   ConfigLoader --> EngineRunner: RunConfig

   EngineRunner -> MarketDataSource: get_snapshot(as_of)
   MarketDataSource --> EngineRunner: MarketDataEnvironment

   EngineRunner -> SimulationDriver: run(models, corr, time_grid, sim_cfg)
   SimulationDriver --> EngineRunner: RiskFactorCube

   EngineRunner -> PortfolioPricer: price_on_cube(portfolio, RiskFactorCube, PricingContext)
   PortfolioPricer --> EngineRunner: ExposureCube (uncollateralised)

   EngineRunner -> CollateralEngine: apply_csa(ExposureCube, CSA)
   CollateralEngine --> EngineRunner: ExposureCube (collateralised)

   EngineRunner -> ExposureMetrics: compute_EE/EPE/EEPE/PFE
   ExposureMetrics --> EngineRunner: ExposureMetricsResults

   EngineRunner -> XVAEngine: compute_XVA(CollateralisedCube, PD/LGD/etc.)
   XVAEngine --> EngineRunner: XVAResults

   EngineRunner -> Backtest: run()
   Backtest --> EngineRunner: BacktestResults

   EngineRunner -> ReportExporter: export_exposure/exposure_metrics/xva(...)
   ReportExporter --> EngineRunner: Files / Reports

   EngineRunner --> User: RunSummary

   @enduml
