# src/test_data_loader.py
from data_loader import DataLoader
from pathlib import Path

def main():
    # Initialize with correct paths
    base_dir = Path(__file__).parent.parent
    loader = DataLoader(
        data_path=base_dir / 'data',
        output_path=base_dir / 'outputs'
    )
    
    # Process all data
    processed_data = loader.process_all_data(
        store_file='stores.csv',
        sales_file='sales.csv'  # Change if your sales file has different name
    )
    
    if processed_data is not None:
        # Display summary
        print("\n📊 Processing Summary:")
        print(f"- Total records: {len(processed_data)}")
        print(f"- Unique stores: {processed_data['Store'].nunique()}")
        
        if 'Date' in processed_data.columns:
            print(f"- Date range: {processed_data['Date'].min()} to {processed_data['Date'].max()}")
            print("- Time features added: DayOfWeek, DayName, Month, Year, IsWeekend, IsHoliday")
        
        # Save results
        loader.save_results(processed_data, 'cleaned_sales_data.csv')
    else:
        print("❌ Failed to process data")

if __name__ == "__main__":
    main()