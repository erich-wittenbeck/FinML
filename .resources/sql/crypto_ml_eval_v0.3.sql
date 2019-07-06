-- --------------------------------------------------------
-- Host:                         127.0.0.1
-- Server Version:               5.7.18-log - MySQL Community Server (GPL)
-- Server Betriebssystem:        Win64
-- HeidiSQL Version:             9.4.0.5125
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;


-- Exportiere Datenbank Struktur für crypto_ml_eval
CREATE DATABASE IF NOT EXISTS `crypto_ml_eval` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `crypto_ml_eval`;

-- Exportiere Struktur von Tabelle crypto_ml_eval.f1_scores
CREATE TABLE IF NOT EXISTS `f1_scores` (
  `uuid` varchar(50) NOT NULL,
  `pos` double NOT NULL,
  `ntr` double NOT NULL,
  `neg` double NOT NULL,
  PRIMARY KEY (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

-- Daten Export vom Benutzer nicht ausgewählt
-- Exportiere Struktur von Tabelle crypto_ml_eval.features
CREATE TABLE IF NOT EXISTS `features` (
  `uuid` varchar(50) NOT NULL,
  `features_list` varchar(1024) NOT NULL,
  PRIMARY KEY (`uuid`),
  UNIQUE KEY `unique_key` (`features_list`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- Daten Export vom Benutzer nicht ausgewählt
-- Exportiere Struktur von Tabelle crypto_ml_eval.precision_scores
CREATE TABLE IF NOT EXISTS `precision_scores` (
  `uuid` varchar(50) NOT NULL,
  `pos` double NOT NULL,
  `ntr` double NOT NULL,
  `neg` double NOT NULL,
  PRIMARY KEY (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- Daten Export vom Benutzer nicht ausgewählt
-- Exportiere Struktur von Tabelle crypto_ml_eval.randf_evaluation
CREATE TABLE IF NOT EXISTS `randf_evaluation` (
  `uuid` varchar(50) NOT NULL,
  `metric` varchar(50) NOT NULL,
  `currency` varchar(50) NOT NULL,
  `candle_size` varchar(50) NOT NULL,
  `n` int(11) NOT NULL DEFAULT '1',
  `margin` double NOT NULL DEFAULT '0',
  `alpha` double NOT NULL DEFAULT '1',
  `features_uuid` varchar(50) NOT NULL,
  `hyperparams_uuid` varchar(50) DEFAULT NULL,
  `precision_scores_uuid` varchar(50) DEFAULT NULL,
  `recall_scores_uuid` varchar(50) DEFAULT NULL,
  `f1_scores_uuid` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`uuid`),
  UNIQUE KEY `unique_params` (`currency`,`candle_size`,`metric`,`n`,`margin`,`alpha`,`features_uuid`),
  KEY `randf_hyperparams_fk` (`hyperparams_uuid`),
  KEY `randf_precision_fk` (`precision_scores_uuid`),
  KEY `randf_recall_fk` (`recall_scores_uuid`),
  KEY `randf_f1_fk` (`f1_scores_uuid`),
  CONSTRAINT `randf_f1_fk` FOREIGN KEY (`f1_scores_uuid`) REFERENCES `f1_scores` (`uuid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `randf_hyperparams_fk` FOREIGN KEY (`hyperparams_uuid`) REFERENCES `randf_hyperparams` (`uuid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `randf_precision_fk` FOREIGN KEY (`precision_scores_uuid`) REFERENCES `precision_scores` (`uuid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `randf_recall_fk` FOREIGN KEY (`recall_scores_uuid`) REFERENCES `recall_scores` (`uuid`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

-- Daten Export vom Benutzer nicht ausgewählt
-- Exportiere Struktur von Tabelle crypto_ml_eval.randf_hyperparams
CREATE TABLE IF NOT EXISTS `randf_hyperparams` (
  `uuid` varchar(50) NOT NULL,
  `n_estimators` int(11) DEFAULT NULL,
  `max_features` int(11) DEFAULT NULL,
  `min_samples_split` int(11) DEFAULT NULL,
  `min_samples_leaf` int(11) DEFAULT NULL,
  `bootstrap` bit(1) DEFAULT NULL,
  PRIMARY KEY (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- Daten Export vom Benutzer nicht ausgewählt
-- Exportiere Struktur von Tabelle crypto_ml_eval.recall_scores
CREATE TABLE IF NOT EXISTS `recall_scores` (
  `uuid` varchar(50) NOT NULL,
  `pos` double NOT NULL,
  `ntr` double NOT NULL,
  `neg` double NOT NULL,
  PRIMARY KEY (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

-- Daten Export vom Benutzer nicht ausgewählt
-- Exportiere Struktur von Tabelle crypto_ml_eval.svm_evaluation
CREATE TABLE IF NOT EXISTS `svm_evaluation` (
  `uuid` varchar(50) NOT NULL,
  `metric` varchar(50) NOT NULL,
  `currency` varchar(50) NOT NULL,
  `candle_size` varchar(50) NOT NULL,
  `n` int(11) NOT NULL DEFAULT '1',
  `margin` double NOT NULL DEFAULT '0',
  `alpha` double NOT NULL DEFAULT '1',
  `features_uuid` varchar(50) NOT NULL,
  `hyperparams_uuid` varchar(50) DEFAULT NULL,
  `precision_scores_uuid` varchar(50) DEFAULT NULL,
  `recall_scores_uuid` varchar(50) DEFAULT NULL,
  `f1_scores_uuid` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`uuid`),
  UNIQUE KEY `svm_features_fk` (`features_uuid`),
  UNIQUE KEY `unique_params` (`metric`,`n`,`margin`,`alpha`,`candle_size`,`currency`),
  KEY `svm_f1_fk` (`f1_scores_uuid`),
  KEY `svm_precision_fk` (`precision_scores_uuid`),
  KEY `svm_recall_fk` (`recall_scores_uuid`),
  KEY `svm_hyperparams_fk` (`hyperparams_uuid`),
  CONSTRAINT `svm_f1_fk` FOREIGN KEY (`f1_scores_uuid`) REFERENCES `f1_scores` (`uuid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `svm_hyperparams_fk` FOREIGN KEY (`hyperparams_uuid`) REFERENCES `svm_hyperparams` (`uuid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `svm_precision_fk` FOREIGN KEY (`precision_scores_uuid`) REFERENCES `precision_scores` (`uuid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `svm_recall_fk` FOREIGN KEY (`recall_scores_uuid`) REFERENCES `recall_scores` (`uuid`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- Daten Export vom Benutzer nicht ausgewählt
-- Exportiere Struktur von Tabelle crypto_ml_eval.svm_hyperparams
CREATE TABLE IF NOT EXISTS `svm_hyperparams` (
  `uuid` varchar(50) NOT NULL,
  `kernel` varchar(50) DEFAULT NULL,
  `c` double DEFAULT NULL,
  `gamma` double DEFAULT NULL,
  `tol` double DEFAULT NULL,
  `shrinking` bit(1) DEFAULT b'0',
  PRIMARY KEY (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- Daten Export vom Benutzer nicht ausgewählt
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IF(@OLD_FOREIGN_KEY_CHECKS IS NULL, 1, @OLD_FOREIGN_KEY_CHECKS) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
